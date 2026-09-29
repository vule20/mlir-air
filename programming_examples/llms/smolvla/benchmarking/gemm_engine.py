# SPDX-License-Identifier: MIT
"""Several bfp16-weight GEMMs under ONE device configuration.

Each air.launch costs a reset plus a full reconfiguration (core programs,
DMA/lock/switch setup): ~63 us + 1.53 us per KB of control code
(reconfig_probe.py). The GEMM engine runs a list of GEMM jobs, one after the
other, inside one launch and one segment: the herd is the same cores with the
same program, and every job reuses the same accumulator, drain and output tile
buffers, so only loop trip counts and DRAM addresses change between jobs.

All jobs share tile_m / tile_n / tile_k_l1 / tile_k_l2 (one microkernel
object, identical memtile DMA patterns); each job has its own K and N.
Args: A_j, B_j (packed), C_j per job, in order.

--mode stitched is the baseline (one launch per job in one ELF); --mode loads
is the first per-job ops.load version, kept because it shows why the channel
version is needed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE.parent.parent), str(_HERE.parent.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from air import api as air
from air.api import ops
from air.api.types import bf16, f32, i8


def build_gemm_engine(m, jobs, tile_m, tile_n, tile_k_l1, herd_m, herd_n, sym_suffix, link_with):
    """jobs: [(k, n, tile_k_l2)] with one tile_k_l2 for all jobs.

    Every DMA is an explicit channel shared by all jobs. With per-job ops.load /
    ops.store each job gets its own channels, and two jobs overflow the memtile's
    48 BDs; air-fuse-channels' aggressive mode, which should time-multiplex them,
    fails to verify ("operand does not dominate this use"). Row r's A tile,
    column c's B tile and row r's output tile each have their own L2 buffer, so
    a memtile carries one A, one B and one C stream, the same as one launch of
    build_gemm_bfp16. The memtile side is one flat loop over every job's K
    steps; only the core loop trip counts and the shim addresses differ by job.
    """
    from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import bfp_tile_bytes

    r, s, t = 8, 8, 8
    tile_bytes = bfp_tile_bytes(tile_n, tile_k_l1)
    l2_m, l2_n = tile_m * herd_m, tile_n * herd_n
    tk2 = jobs[0][2]
    assert all(j[2] == tk2 for j in jobs), "the channel engine needs one tile_k_l2"
    k_per_l2 = tk2 // tile_k_l1
    assert m % l2_m == 0 and tk2 % tile_k_l1 == 0
    tensors = []
    for k, n, _ in jobs:
        assert n % l2_n == 0 and k % tk2 == 0
        tensors.append((
            air.tensor([m, k], bf16),
            air.tensor([n // tile_n, k // tile_k_l1, tile_bytes], i8),
            air.tensor([m, n], bf16),
        ))
    n_tiles = [(m // l2_m) * (n // l2_n) for _, n, _ in jobs]
    k_steps = sum(nt * (k // tk2) for nt, (k, _, _) in zip(n_tiles, jobs))

    zero_acc = air.extern(f"zero_vectorized_f32_mn{sym_suffix}", link_with=link_with)
    matmul = air.extern(f"matmul_bf16_x_bfp16_packed_f32{sym_suffix}", link_with=link_with)
    drain_fn = air.extern(f"f32_to_bf16_mn{sym_suffix}", link_with=link_with)

    a_in = air.channel("EngAIn", size=[herd_m])
    b_in = air.channel("EngBIn", size=[herd_n])
    a2l1 = air.channel("EngA2L1", size=[herd_m, 1], broadcast_shape=[herd_m, herd_n])
    b2l1 = air.channel("EngB2L1", size=[1, herd_n], broadcast_shape=[herd_m, herd_n])
    c2l2 = air.channel("EngC2L2", size=[herd_m, herd_n])
    c_out = air.channel("EngCOut", size=[herd_m])

    with air.launch(name="gemm_engine") as launch:

        @launch.body
        def _():
            with air.segment(name="engine_seg") as seg:

                @seg.body
                def _():
                    for (k, n, _), (A, B, C) in zip(jobs, tensors):
                        for li in air.sequential(m // l2_m):
                            for lj in air.sequential(n // l2_n):
                                for k2 in air.sequential(k // tk2):
                                    for i in range(herd_m):
                                        row = li * l2_m + i * tile_m
                                        a_in.put(A[row : row + tile_m, k2 * tk2 : k2 * tk2 + tk2], indices=[i])
                                    for j in range(herd_n):
                                        kc = k2 * k_per_l2
                                        b_in.put(B[lj * herd_n + j, kc : kc + k_per_l2, :], indices=[j])

                    def l2(shape, dtype, col):
                        return air.alloc(shape, dtype, scope=seg.private(), column=col, split=False)

                    l2_a = [l2([tile_m, tk2], bf16, i) for i in range(herd_m)]
                    l2_b = [l2([k_per_l2, tile_bytes], i8, j) for j in range(herd_n)]
                    l2_c = [l2([tile_m, l2_n], bf16, i) for i in range(herd_m)]

                    for i in range(herd_m):
                        for _ in air.sequential(k_steps):
                            a_in.get(l2_a[i], indices=[i])
                            for j in range(k_per_l2):
                                a2l1.put(
                                    l2_a[i][:, j * tile_k_l1 : (j + 1) * tile_k_l1]
                                    .reshape(1, 1, tile_m // r, r, tile_k_l1 // s, s)
                                    .transpose(0, 1, 2, 4, 3, 5),
                                    indices=[i, 0],
                                )
                    for j in range(herd_n):
                        for _ in air.sequential(k_steps):
                            b_in.get(l2_b[j], indices=[j])
                            for jj in range(k_per_l2):
                                b2l1.put(l2_b[j][jj, :], indices=[0, j])

                    with air.herd([range(herd_m), range(herd_n)], name="herd_0",
                                  shape=(herd_m, herd_n)) as h:

                        @h.body
                        def _(tx, ty):
                            acc = air.alloc([1, 1, tile_n // t, tile_m // r, r, t], f32, scope=h.private())
                            drain = air.alloc([1, 1, tile_n // t, tile_m // r, r, t], bf16, scope=h.private())
                            l1_a = air.alloc([1, 1, tile_m // r, tile_k_l1 // s, r, s], bf16, scope=h.private())
                            l1_b = air.alloc([tile_bytes], i8, scope=h.private())
                            for (k, _, _), nt in zip(jobs, n_tiles):
                                for _ in air.sequential(nt):
                                    zero_acc(acc)
                                    for _ in air.sequential(k // tile_k_l1):
                                        a2l1.get(l1_a, indices=[tx, ty])
                                        b2l1.get(l1_b, indices=[tx, ty])
                                        matmul(l1_a, l1_b, acc)
                                    drain_fn(acc, drain)
                                    c2l2.put(drain.transpose(0, 1, 3, 4, 2, 5), indices=[tx, ty])

                    for i in range(herd_m):
                        for _ in air.sequential(sum(n_tiles)):
                            for j in air.parallel(0, herd_n):
                                c2l2.get(l2_c[i][:, j * tile_n : (j + 1) * tile_n], indices=[i, j])
                            c_out.put(l2_c[i], indices=[i])

                    for (k, n, _), (A, B, C) in zip(jobs, tensors):
                        for li in air.sequential(m // l2_m):
                            for lj in air.sequential(n // l2_n):
                                for i in range(herd_m):
                                    row = li * l2_m + i * tile_m
                                    c_out.get(C[row : row + tile_m, lj * l2_n : lj * l2_n + l2_n], indices=[i])

    return launch.build(target="npu2")


def build_gemm_engine_loads(m, jobs, tile_m, tile_n, tile_k_l1, herd_m, herd_n, sym_suffix, link_with):
    """First attempt, ops.load/ops.store per job: one job compiles (same time as
    one launch); two jobs overflow the memtile BDs. jobs: [(k, n, tile_k_l2)]."""
    from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import bfp_tile_bytes

    r, s, t = 8, 8, 8
    tile_bytes = bfp_tile_bytes(tile_n, tile_k_l1)
    l2_m, l2_n = tile_m * herd_m, tile_n * herd_n
    assert m % l2_m == 0
    tensors = []
    for k, n, tk2 in jobs:
        assert n % l2_n == 0 and k % tk2 == 0 and tk2 % tile_k_l1 == 0
        tensors.append((
            air.tensor([m, k], bf16),
            air.tensor([n // tile_n, k // tile_k_l1, tile_bytes], i8),
            air.tensor([m, n], bf16),
        ))

    zero_acc = air.extern(f"zero_vectorized_f32_mn{sym_suffix}", link_with=link_with)
    matmul = air.extern(f"matmul_bf16_x_bfp16_packed_f32{sym_suffix}", link_with=link_with)
    drain_fn = air.extern(f"f32_to_bf16_mn{sym_suffix}", link_with=link_with)

    with air.launch(name="gemm_engine") as launch:

        @launch.body
        def _():
            with air.segment(name="engine_seg") as seg:

                @seg.body
                def _():
                    hd = (herd_m, herd_n)
                    acc = air.alloc([*hd, tile_n // t, tile_m // r, r, t], f32, scope=seg.shared())
                    drain = air.alloc([*hd, tile_n // t, tile_m // r, r, t], bf16, scope=seg.shared())
                    l2_c = air.alloc([herd_m, herd_n, tile_m, tile_n], bf16, scope=seg.private())

                    def herd():
                        return air.herd([range(hd[0]), range(hd[1])], name="herd_0", shape=hd)

                    def l2_inputs(tk2):
                        return (air.alloc([herd_m, tile_m, tk2], bf16, scope=seg.private()),
                                air.alloc([herd_n, tk2 // tile_k_l1, tile_bytes], i8, scope=seg.private()))

                    # With one tile_k_l2 for every job, the L2 input buffers and hence every
                    # memtile DMA pattern are identical across jobs; only trip counts differ.
                    shared_l2 = l2_inputs(jobs[0][2]) if len({j[2] for j in jobs}) == 1 else None

                    for (k, n, tk2), (A, B, C) in zip(jobs, tensors):
                        k_per_l2 = tk2 // tile_k_l1
                        l2_a, l2_b = shared_l2 or l2_inputs(tk2)
                        for li in air.sequential(m // l2_m):
                            for lj in air.sequential(n // l2_n):
                                row, col = li * l2_m, lj * l2_n
                                n_outer = lj * herd_n

                                with herd() as zh:

                                    @zh.body
                                    def _(tx, ty):
                                        zero_acc(acc)

                                for k2 in air.sequential(k // tk2):
                                    k_l2_off = k2 * tk2
                                    k_chunk_off = k2 * k_per_l2
                                    ops.load(
                                        l2_a,
                                        A[row : row + l2_m, k_l2_off : k_l2_off + tk2].reshape(herd_m, tile_m, tk2),
                                    )
                                    ops.load(l2_b, B[n_outer : n_outer + herd_n, k_chunk_off : k_chunk_off + k_per_l2, :])

                                    with herd() as h:

                                        @h.body
                                        def _(tx, ty):
                                            l1_a = air.alloc([1, 1, tile_m // r, tile_k_l1 // s, r, s], bf16,
                                                             scope=h.private())
                                            l1_b = air.alloc([tile_bytes], i8, scope=h.private())
                                            for j in air.sequential(k_per_l2):
                                                k1 = j * tile_k_l1
                                                ops.load(
                                                    l1_a,
                                                    l2_a[tx, :, k1 : k1 + tile_k_l1]
                                                    .reshape(1, 1, tile_m // r, r, tile_k_l1 // s, s)
                                                    .transpose(0, 1, 2, 4, 3, 5),
                                                )
                                                ops.load(l1_b, l2_b[ty, j, :])
                                                matmul(l1_a, l1_b, acc)

                                with herd() as dh:

                                    @dh.body
                                    def _(tx, ty):
                                        drain_fn(acc, drain)
                                        ops.store(drain[tx, ty, :, :, :, :].transpose(0, 1, 3, 4, 2, 5),
                                                  l2_c[tx, ty, :, :])

                                ops.store(l2_c.transpose(0, 2, 1, 3), C[row : row + l2_m, col : col + l2_n])

    return launch.build(target="npu2")


def build_stitched(m, jobs, tile_m, tile_n, tile_k_l1, herd_m, herd_n, sym_suffix, link_with):
    """Baseline: the same jobs as separate launches (one configuration each) in one ELF."""
    from gemm_bfp16 import bfp16_extern_syms, bfp16_weight_type, build_gemm_bfp16
    from shared.infra.stitching import FuncArg, KernelSlice, stitch_elf

    args, slices = [], []
    for i, (k, n, tk2) in enumerate(jobs):
        base = len(args)
        args += [FuncArg(f"%arg{base}", f"memref<{m}x{k}xbf16>"),
                 FuncArg(f"%arg{base + 1}", bfp16_weight_type(k, n, tile_n, tile_k_l1)),
                 FuncArg(f"%arg{base + 2}", f"memref<{m}x{n}xbf16>")]
        ir = str(build_gemm_bfp16(m, k, n, tile_m, tk2, tile_k_l1, tile_n, herd_m, herd_n, sym_suffix, link_with))
        slices.append(KernelSlice(ir, f"g{i}", {0: base, 1: base + 1, 2: base + 2},
                                  extern_syms=bfp16_extern_syms(sym_suffix), private_from=(i == 0)))
    return stitch_elf("gemm_stitched", args, slices)


SHAPES = {"qkv": (960, 1600), "o": (960, 960), "gu": (960, 5120), "dn": (2560, 960)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="o,dn", help="comma list of GEMM shapes")
    ap.add_argument("--tk2", default="480,320", help="tile_k_l2 per job")
    ap.add_argument("--tk1", type=int, default=160)
    ap.add_argument("--tile-n", type=int, default=80)
    ap.add_argument("--mode", default="engine", choices=["engine", "loads", "stitched"])
    ap.add_argument("--pingpong", default="", help="omit_pingpong value")
    ap.add_argument("--tiling", default="2,3", help="stitched launches' runtime_loop_tiling_sizes")
    ap.add_argument("--chmux", default="", help="air channel multiplexing memory spaces, e.g. L2 or L1,L2")
    ap.add_argument("--debug-ir", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    from gemm_bfp16 import compile_mm_bfp16
    from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import pack_b_bfp16ebs8
    from shared.infra.cache import KernelCache, Profiler

    m, tile_m, herd = 256, 32, 4
    names = args.jobs.split(",")
    jobs = [(*SHAPES[nm], int(t)) for nm, t in zip(names, args.tk2.split(","))]
    tag = (f"{args.mode}_{'-'.join(names)}_k{args.tk2.replace(',', '-')}x{args.tk1}_n{args.tile_n}"
           f"{'_pp' + args.pingpong if args.pingpong else ''}"
           f"{'_mux' + args.chmux.replace(',', '') if args.chmux else ''}")
    cache = KernelCache(str(_HERE / "build" / f"gemm_engine_{tag}"), verbose=False, profiler=Profiler(enabled=True))
    sfx, obj = "_eng", "mm_eng.o"
    compile_mm_bfp16(tile_m, args.tile_n, args.tk1, sfx, obj)
    build = {"engine": build_gemm_engine, "loads": build_gemm_engine_loads, "stitched": build_stitched}[args.mode]
    mod = build(m, jobs, tile_m, args.tile_n, args.tk1, herd, herd, sfx, obj)
    backend = {"verbose": False, "omit_while_true_loop": False, "output_format": "elf",
               "instance_name": "gemm_stitched" if args.mode == "stitched" else "gemm_engine",
               "omit_pingpong": args.pingpong,
               "channel_multiplexing": args.chmux.split(",") if args.chmux else [],
               "debug_ir": args.debug_ir}
    if args.mode == "stitched":
        backend["runtime_loop_tiling_sizes"] = [int(v) for v in args.tiling.split(",")]
    cache.compile_and_cache("gemm", mod, backend)

    rng = np.random.default_rng(0)
    bufs, outs, refs = [], [], []
    for k, n, _ in jobs:
        a = (rng.standard_normal((m, k)) * 0.5).astype(bfloat16)
        w = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(bfloat16)
        bufs += [a, pack_b_bfp16ebs8(w, args.tile_n, args.tk1), np.zeros((m, n), bfloat16)]
        outs.append(len(bufs) - 1)
        refs.append(a.astype(np.float32) @ w.astype(np.float32))

    def run():
        return cache.load_and_run("gemm", backend, *bufs, output_indices=outs, bo_key="g")

    res = run()
    for i, (nm, (k, n, _)) in enumerate(zip(names, jobs)):
        o = np.asarray(res[outs[i]], dtype=np.float32).reshape(m, n)
        ref = refs[i]
        cos = float(o.ravel() @ ref.ravel() / (np.linalg.norm(o) * np.linalg.norm(ref)))
        np.save(_HERE / "build" / f"gemm_engine_{tag}_{nm}.npy", o)
        print(f"  {nm}: cosine {cos:.6f}")
    cache.profiler.kernel_breakdowns.clear()
    for _ in range(args.iters):
        run()
    dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["gemm"])
    print(f"{tag}: device median {dev[len(dev) // 2] * 1e3:.0f} us (min {dev[0] * 1e3:.0f}, "
          f"p10 {dev[len(dev) // 10] * 1e3:.0f})")


if __name__ == "__main__":
    main()
