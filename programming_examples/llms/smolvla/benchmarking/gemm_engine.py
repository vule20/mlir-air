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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE.parent.parent), str(_HERE.parent.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from air import api as air
from air.api import ops
from air.api.types import bf16, f32, i32, i8


@dataclass
class Job:
    """One engine GEMM job, C = epilogue(A @ B), over named tensors.

    A [m, k] bf16, B packed bfp16 [n / tile_n, k / tile_k_l1, bytes], C [m, n_out].
    residual: C += R ([m, n_out]) in f32 at drain; R's tiles ride the A channel as
      one extra tile_k_l2 step per output tile (needs tile_k_l2 == l2_n).
    rms: rows of A are RMS-normalised: the norm weight is folded into B on the
      host, the per-row sum of squares is accumulated from the A chunks and the
      drain scales rows by rsqrt(ss / k + eps) (kernel RMS_K must equal k).
    swiglu: B's tile_n blocks hold tile_n/2 gate then tile_n/2 up columns; two
      consecutive output tiles fill the two halves of one tile_n-wide store, so
      n_out = n / 2 (B columns permuted on the host, see permute_gate_up).
    rope: RoPE table P ([m, n] bf16, (cos, sin) per adjacent column pair, (1, 0)
      to pass a pair through), riding the A channel like a residual. The rotation
      acts on adjacent pairs, so head dims must be pair-interleaved in B (see
      interleave_rope_heads). Needs rms (the scale is applied first).
    """

    a: str
    b: str
    c: str
    k: int
    n: int
    residual: str = None
    rms: bool = False
    swiglu: bool = False
    rope: str = None

    @property
    def n_out(self):
        return self.n // 2 if self.swiglu else self.n


def build_gemm_engine(m, jobs, tile_m, tile_n, tile_k_l1, tile_k_l2, herd_m, herd_n, sym_suffix, link_with,
                      arg_order=None):
    """jobs: [Job]. Args are the jobs' named tensors, in arg_order (default:
    first appearance, A, B, residual, C per job).

    Every DMA is an explicit channel shared by all jobs. With per-job ops.load /
    ops.store each job gets its own channels, and two jobs overflow the memtile's
    48 BDs; air-fuse-channels' aggressive mode, which should time-multiplex them,
    fails to verify ("operand does not dominate this use"). Row r's A tile,
    column c's B tile and row r's output tile each have their own L2 buffer, so
    a memtile carries one A, one B and one C stream, the same as one launch of
    build_gemm_bfp16. The memtile side is one flat loop over every job's K
    steps; only the core loop trip counts and the shim addresses differ by job.
    A core DMA channel cycles one BD chain, so every transfer into or out of a
    core has the same size in every job: residual tiles come in as A chunks and
    SwiGLU halves are paired into full-width output tiles.
    """
    from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import bfp_tile_bytes

    r, s, t = 8, 8, 8
    tile_bytes = bfp_tile_bytes(tile_n, tile_k_l1)
    l2_m, l2_n = tile_m * herd_m, tile_n * herd_n
    tk2 = tile_k_l2
    k_per_l2 = tk2 // tile_k_l1
    assert m % l2_m == 0 and tk2 % tile_k_l1 == 0

    shapes = {}

    def declare(name, shape, dtype):
        assert shapes.setdefault(name, (shape, dtype)) == (shape, dtype), (name, shapes[name], shape)

    for j in jobs:
        assert j.n % l2_n == 0 and j.k % tk2 == 0
        assert not (j.residual or j.rope) or tk2 == l2_n, "a residual tile is one tile_k_l2 step"
        assert not (j.residual and j.rope) and (not j.rope or (j.rms and not j.swiglu))
        assert not j.swiglu or (j.n // l2_n) % 2 == 0
        declare(j.a, [m, j.k], bf16)
        declare(j.b, [j.n // tile_n, j.k // tile_k_l1, tile_bytes], i8)
        if j.residual:
            declare(j.residual, [m, j.n_out], bf16)
        if j.rope:
            declare(j.rope, [m, j.n], bf16)
        declare(j.c, [m, j.n_out], bf16)
    arg_order = arg_order or list(shapes)
    assert sorted(arg_order) == sorted(shapes), (arg_order, list(shapes))
    T = {name: air.tensor(*shapes[name]) for name in arg_order}

    n_tiles = [(m // l2_m) * (j.n // l2_n) for j in jobs]
    a_steps = sum(nt * (j.k // tk2 + bool(j.residual or j.rope)) for nt, j in zip(n_tiles, jobs))
    b_steps = sum(nt * (j.k // tk2) for nt, j in zip(n_tiles, jobs))
    c_tiles = sum(nt // (2 if j.swiglu else 1) for nt, j in zip(n_tiles, jobs))

    def ext(name, **kw):
        return air.extern(f"{name}{sym_suffix}", link_with=link_with, **kw)

    zero_acc = ext("zero_vectorized_f32_mn")
    matmul = ext("matmul_bf16_x_bfp16_packed_f32")
    drain_fn = ext("f32_to_bf16_mn")
    if any(j.residual for j in jobs):
        add_res = ext("add_residual_blocked", scalars=[i32])
    if any(j.rope for j in jobs):
        rope_fn = ext("rms_rope_blocked", scalars=[i32])
    if any(j.rms for j in jobs):
        zero_rows = ext("zero_rows")
        sumsq = ext("sumsq_rows_blocked")
        rows_rstd = ext("rows_rstd")
    if any(j.swiglu for j in jobs):
        swiglu_fn = ext("f32_to_bf16_rms_swiglu" if all(j.rms for j in jobs if j.swiglu) else "f32_to_bf16_swiglu_mn",
                        scalars=[i32])
        assert all(j.rms == jobs[[x.swiglu for x in jobs].index(True)].rms for j in jobs if j.swiglu)
    assert all(not j.rms or j.swiglu or j.rope for j in jobs), "rms is only in the SwiGLU and RoPE drains"

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
                    for j in jobs:
                        A, B = T[j.a], T[j.b]
                        # air-isolate-async-dma-loop-nests gives every put its own loop nest,
                        # which would send all of a job's A steps before any residual step.
                        # Unrolled tile loops keep each tile's residual put right after its K steps.
                        tile_loop = range if j.residual or j.rope else air.sequential
                        for li in tile_loop(m // l2_m):
                            for lj in tile_loop(j.n // l2_n):
                                for k2 in air.sequential(j.k // tk2):
                                    for i in range(herd_m):
                                        row = li * l2_m + i * tile_m
                                        a_in.put(A[row : row + tile_m, k2 * tk2 : k2 * tk2 + tk2], indices=[i])
                                # The shim command stream zips the channels' task lists, and a BD-reuse
                                # await on a later job's task before this job's last A task deadlocks.
                                # A residual tile is two A tasks (K steps, residual), so B is two too.
                                k_steps = j.k // tk2
                                halves = ([(0, k_steps // 2), (k_steps // 2, k_steps)] if j.residual or j.rope
                                          else [(0, k_steps)])
                                for lo, hi in halves:
                                    for k2 in air.sequential(lo, hi):
                                        for c in range(herd_n):
                                            kc = k2 * k_per_l2
                                            b_in.put(B[lj * herd_n + c, kc : kc + k_per_l2, :], indices=[c])
                                if j.residual or j.rope:
                                    R = T[j.residual or j.rope]
                                    for i in range(herd_m):
                                        row = li * l2_m + i * tile_m
                                        a_in.put(R[row : row + tile_m, lj * l2_n : lj * l2_n + l2_n], indices=[i])
                        # Right after the job's own puts: a later job that reads C then waits on
                        # transfers issued before it (emitted at the end, O -> Down's residual hangs).
                        C = T[j.c]
                        for li in air.sequential(m // l2_m):
                            for lj in air.sequential(j.n_out // l2_n):
                                for i in range(herd_m):
                                    row = li * l2_m + i * tile_m
                                    c_out.get(C[row : row + tile_m, lj * l2_n : lj * l2_n + l2_n], indices=[i])

                    def l2(shape, dtype, col):
                        return air.alloc(shape, dtype, scope=seg.private(), column=col, split=False)

                    l2_a = [l2([tile_m, tk2], bf16, i) for i in range(herd_m)]
                    l2_b = [l2([k_per_l2, tile_bytes], i8, c) for c in range(herd_n)]
                    l2_c = [l2([tile_m, l2_n], bf16, i) for i in range(herd_m)]

                    for i in range(herd_m):
                        for _ in air.sequential(a_steps):
                            a_in.get(l2_a[i], indices=[i])
                            for kk in range(k_per_l2):
                                a2l1.put(
                                    l2_a[i][:, kk * tile_k_l1 : (kk + 1) * tile_k_l1]
                                    .reshape(1, 1, tile_m // r, r, tile_k_l1 // s, s)
                                    .transpose(0, 1, 2, 4, 3, 5),
                                    indices=[i, 0],
                                )
                    for c in range(herd_n):
                        for _ in air.sequential(b_steps):
                            b_in.get(l2_b[c], indices=[c])
                            for kk in range(k_per_l2):
                                b2l1.put(l2_b[c][kk, :], indices=[0, c])

                    with air.herd([range(herd_m), range(herd_n)], name="herd_0",
                                  shape=(herd_m, herd_n)) as h:

                        @h.body
                        def _(tx, ty):
                            acc = air.alloc([1, 1, tile_n // t, tile_m // r, r, t], f32, scope=h.private())
                            drain = air.alloc([1, 1, tile_n // t, tile_m // r, r, t], bf16, scope=h.private())
                            l1_a = air.alloc([1, 1, tile_m // r, tile_k_l1 // s, r, s], bf16, scope=h.private())
                            l1_b = air.alloc([tile_bytes], i8, scope=h.private())
                            ss = (air.alloc([tile_m], f32, scope=h.private()) if any(j.rms for j in jobs)
                                  else None)

                            def tile(j, half=None, stats=False):
                                zero_acc(acc)
                                if stats:
                                    zero_rows(ss)
                                for _ in air.sequential(j.k // tile_k_l1):
                                    a2l1.get(l1_a, indices=[tx, ty])
                                    b2l1.get(l1_b, indices=[tx, ty])
                                    if stats:
                                        sumsq(l1_a, ss)
                                    matmul(l1_a, l1_b, acc)
                                if stats:
                                    rows_rstd(ss)
                                if j.residual or j.rope:
                                    # Core column c's output columns are A chunk c // 2, half c % 2.
                                    for ch in range(k_per_l2):
                                        a2l1.get(l1_a, indices=[tx, ty])
                                        for hf in range(tile_k_l1 // tile_n):
                                            with ops.branch(ty == ch * (tile_k_l1 // tile_n) + hf):
                                                if j.rope:
                                                    rope_fn(acc, ss, l1_a, hf)
                                                else:
                                                    add_res(acc, l1_a, hf)
                                if j.swiglu:
                                    if j.rms:
                                        swiglu_fn(acc, ss, drain, half)
                                    else:
                                        swiglu_fn(acc, drain, half)
                                else:
                                    drain_fn(acc, drain)

                            for j, nt in zip(jobs, n_tiles):
                                if j.swiglu and j.rms:
                                    # The row statistics depend on the row block only: gathered on its
                                    # first tile's K pass, reused by the rest (tiles are li-major).
                                    pairs = j.n // l2_n // 2
                                    for _ in air.sequential(m // l2_m):
                                        tile(j, 0, stats=True)
                                        tile(j, 1)
                                        c2l2.put(drain.transpose(0, 1, 3, 4, 2, 5), indices=[tx, ty])
                                        for _ in air.sequential(pairs - 1):
                                            tile(j, 0)
                                            tile(j, 1)
                                            c2l2.put(drain.transpose(0, 1, 3, 4, 2, 5), indices=[tx, ty])
                                elif j.rms:
                                    for _ in air.sequential(m // l2_m):
                                        tile(j, stats=True)
                                        c2l2.put(drain.transpose(0, 1, 3, 4, 2, 5), indices=[tx, ty])
                                        for _ in air.sequential(j.n // l2_n - 1):
                                            tile(j)
                                            c2l2.put(drain.transpose(0, 1, 3, 4, 2, 5), indices=[tx, ty])
                                elif j.swiglu:
                                    for _ in air.sequential(nt // 2):
                                        tile(j, 0)
                                        tile(j, 1)
                                        c2l2.put(drain.transpose(0, 1, 3, 4, 2, 5), indices=[tx, ty])
                                else:
                                    for _ in air.sequential(nt):
                                        tile(j)
                                        c2l2.put(drain.transpose(0, 1, 3, 4, 2, 5), indices=[tx, ty])

                    for i in range(herd_m):
                        for _ in air.sequential(c_tiles):
                            for c in air.parallel(0, herd_n):
                                c2l2.get(l2_c[i][:, c * tile_n : (c + 1) * tile_n], indices=[i, c])
                            c_out.put(l2_c[i], indices=[i])


    return launch.build(target="npu2")


def permute_gate_up(w_gate, w_up, tile_n, l2_n):
    """(K, H) gate/up -> (K, 2H) B for Job(swiglu=True): output tile pair p, half h,
    core column c holds output columns p*l2_n + c*tile_n + h*tile_n/2 + [0, tile_n/2),
    computed from B tile (2p + h) * (l2_n / tile_n) + c = [gate cols | up cols]."""
    k, hdim = w_gate.shape
    half, cols = tile_n // 2, l2_n // tile_n
    o = np.arange(hdim)
    p, c, hh, i = o // l2_n, (o % l2_n) // tile_n, (o % tile_n) // half, o % half
    nt = (2 * p + hh) * cols + c
    w = np.empty((k, 2 * hdim), dtype=w_gate.dtype)
    w[:, nt * tile_n + i] = w_gate
    w[:, nt * tile_n + half + i] = w_up
    return w


def rope_pair_perm(n_heads, head_dim):
    """Column order pair-interleaving each head: new columns 2p, 2p+1 of a head
    are its old p and p + head_dim/2 (the rotate-half partners)."""
    h = head_dim // 2
    local = np.stack([np.arange(h), np.arange(h) + h], axis=1).ravel()
    return (np.arange(n_heads)[:, None] * head_dim + local).ravel()


def qkv_col_perm(n_heads, n_kv_heads, head_dim):
    """B / output column order of the rms+QKV+RoPE job: q and k heads
    pair-interleaved, v unchanged. q.k per head is invariant (same permutation)."""
    q, kv = n_heads * head_dim, n_kv_heads * head_dim
    return np.concatenate([rope_pair_perm(n_heads, head_dim), q + rope_pair_perm(n_kv_heads, head_dim),
                           q + kv + np.arange(kv)])


def rope_table(lut, n_heads, n_kv_heads, head_dim, v_cols):
    """lut [m, head_dim] = [cos | sin] per row -> the Job(rope=) table: (cos, sin)
    per pair-interleaved q and k column pair, (1, 0) over the v columns."""
    h = head_dim // 2
    pair = np.stack([lut[:, :h], lut[:, h:]], axis=2).reshape(len(lut), head_dim)
    v = np.tile(np.array([1.0, 0.0], np.float32), (len(lut), v_cols // 2))
    return np.concatenate([np.tile(pair, (1, n_heads + n_kv_heads)), v], axis=1)


def compile_mm_engine(tile_m, tile_n, tile_k_l1, sym_suffix, out_name, rms_k=960):
    from shared.infra.external_kernels import _PROJ_ROOT, _compile_kernel

    extra = [
        f"-I{_PROJ_ROOT / 'matrix_multiplication' / 'bf16_x_bfp16'}",
        f"-DDIM_M={tile_m}", f"-DDIM_N={tile_n}", f"-DDIM_K={tile_k_l1}",
        f"-DSYM_SUFFIX={sym_suffix}", f"-DRMS_K={rms_k}", "-Wno-macro-redefined",
    ]
    _compile_kernel(_HERE / "kernels_bfp16" / "mm_engine.cc", out_name, extra_flags=extra, force=True)


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
    ap.add_argument("--mode", default="engine", choices=["engine", "loads", "stitched", "ffn", "qkv"])
    ap.add_argument("--pingpong", default="", help="omit_pingpong value")
    ap.add_argument("--tiling", default="2,3", help="stitched launches' runtime_loop_tiling_sizes")
    ap.add_argument("--chmux", default="", help="air channel multiplexing memory spaces, e.g. L2 or L1,L2")
    ap.add_argument("--debug-ir", action="store_true")
    ap.add_argument("--ffn-jobs", default="o,gu,dn", help="--mode ffn: subset of the three jobs")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    if args.mode == "ffn":
        return main_ffn(args)
    if args.mode == "qkv":
        return main_qkv(args)

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
    if args.mode == "engine":
        assert len(set(t for *_, t in jobs)) == 1, "the engine takes one tile_k_l2"
        mod = build_gemm_engine(m, [Job(f"A{i}", f"B{i}", f"C{i}", k, n) for i, (k, n, _) in enumerate(jobs)],
                                tile_m, args.tile_n, args.tk1, jobs[0][2], herd, herd, sfx, obj)
    else:
        build = {"loads": build_gemm_engine_loads, "stitched": build_stitched}[args.mode]
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


def _cos(a, b):
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main_ffn(args):
    """O + residual, RMSNorm + GateUp + SwiGLU, Down + residual as three jobs of one engine launch."""
    from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import pack_b_bfp16ebs8
    from shared.infra.cache import KernelCache, Profiler

    m, emb, hid, tile_m, herd = 256, 960, 2560, 32, 4
    tn, tk1 = args.tile_n, args.tk1
    l2_n = tn * herd
    all_jobs = {
        "o": Job("attn", "wo", "res1", emb, emb, residual="x"),
        "gu": Job("res1", "wgu", "sw", emb, 2 * hid, rms=True, swiglu=True),
        "dn": Job("sw", "wdn", "out", hid, emb, residual="res1"),
    }
    sel = args.ffn_jobs.split(",")
    jobs = [all_jobs[nm] for nm in sel]
    names = ["attn", "wo", "x", "res1", "wgu", "sw", "wdn", "out"]
    used = {getattr(j, f) for j in jobs for f in ("a", "b", "c", "residual")} - {None}
    order = [nm for nm in names if nm in used]
    tag = f"ffn{'' if len(sel) == 3 else '_' + '-'.join(sel)}_n{tn}_k{tk1}{'_pp' + args.pingpong if args.pingpong else ''}"
    cache = KernelCache(str(_HERE / "build" / f"gemm_engine_{tag}"), verbose=False, profiler=Profiler(enabled=True))
    sfx, obj = "_eng", "mm_engine.o"
    compile_mm_engine(tile_m, tn, tk1, sfx, obj, rms_k=emb)
    mod = build_gemm_engine(m, jobs, tile_m, tn, tk1, l2_n, herd, herd, sfx, obj, arg_order=order)
    backend = {"verbose": False, "omit_while_true_loop": False, "output_format": "elf",
               "instance_name": "gemm_engine", "omit_pingpong": args.pingpong, "debug_ir": args.debug_ir}
    cache.compile_and_cache("ffn", mod, backend)

    rng = np.random.default_rng(0)
    f32 = np.float32

    def bf(x):
        return np.asarray(x, f32).astype(bfloat16)

    attn = bf(rng.standard_normal((m, emb)) * 0.5)
    x = bf(rng.standard_normal((m, emb)))
    wo = bf(rng.standard_normal((emb, emb)) / np.sqrt(emb))
    wg = bf(rng.standard_normal((emb, hid)) / np.sqrt(emb))
    wu = bf(rng.standard_normal((emb, hid)) / np.sqrt(emb))
    wd = bf(rng.standard_normal((hid, emb)) / np.sqrt(hid))
    nw = bf(1.0 + 0.1 * rng.standard_normal(emb))
    wgu = permute_gate_up(bf(nw.astype(f32)[:, None] * wg.astype(f32)),
                          bf(nw.astype(f32)[:, None] * wu.astype(f32)), tn, l2_n)
    bufs = [attn, pack_b_bfp16ebs8(wo, tn, tk1), x, np.zeros((m, emb), bfloat16),
            pack_b_bfp16ebs8(wgu, tn, tk1), np.zeros((m, hid), bfloat16),
            pack_b_bfp16ebs8(wd, tn, tk1), np.zeros((m, emb), bfloat16)]
    if "o" not in sel:
        bufs[3] = bf(rng.standard_normal((m, emb)))
    if "gu" not in sel:
        bufs[5] = bf(rng.standard_normal((m, hid)) * 0.3)
    bufs = [b for nm, b in zip(names, bufs) if nm in used]
    outs = [order.index(nm) for nm in ("res1", "sw", "out") if nm in used]

    def silu(v):
        return v / (1.0 + np.exp(-v))

    def ref_res1():
        return attn.astype(f32) @ wo.astype(f32) + x.astype(f32)

    def ref_sw(r1):
        nrm = r1 / np.sqrt(np.mean(r1 * r1, axis=1, keepdims=True) + 1e-5) * nw.astype(f32)
        return silu(nrm @ wg.astype(f32)) * (nrm @ wu.astype(f32))

    def ref_out(s, r1):
        return s @ wd.astype(f32) + r1

    def run():
        return cache.load_and_run("ffn", backend, *bufs, output_indices=outs, bo_key="ffn")

    res = run()
    from reconfig_probe import ctrl_kb
    print(f"  control code {ctrl_kb(cache.cache_dir):.1f} KB")
    if len(sel) < 3:
        for nm in sel:
            if nm == "o":
                print(f"  res1: cosine {_cos(np.asarray(res[order.index('res1')], f32), ref_res1()):.6f}")
            if nm == "dn":
                producer = {"res1": "o", "sw": "gu"}

                def data(v):
                    src = res[order.index(v)] if producer[v] in sel else bufs[order.index(v)]
                    return np.asarray(src, f32).reshape(m, -1)

                sw_in, r1_in = data("sw"), data("res1")
                print(f"  out:  cosine {_cos(np.asarray(res[order.index('out')], f32), ref_out(sw_in, r1_in)):.6f}")
    else:
        r1, sw, out = (np.asarray(res[i], dtype=f32).reshape(m, -1) for i in outs)
        R1 = ref_res1()
        SW = ref_sw(R1)
        print(f"  res1: cosine {_cos(r1, R1):.6f}")
        print(f"  sw:   cosine {_cos(sw, SW):.6f} (from NPU res1: {_cos(sw, ref_sw(r1)):.6f})")
        print(f"  out:  cosine {_cos(out, ref_out(SW, R1)):.6f} (from NPU sw, res1: {_cos(out, ref_out(sw, r1)):.6f})")
        for nm, v in (("res1", r1), ("sw", sw), ("out", out)):
            np.save(_HERE / "build" / f"gemm_engine_{tag}_{nm}.npy", v)
    cache.profiler.kernel_breakdowns.clear()
    for _ in range(args.iters):
        run()
    dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["ffn"])
    print(f"{tag}: device median {dev[len(dev) // 2] * 1e3:.0f} us (min {dev[0] * 1e3:.0f}, "
          f"p10 {dev[len(dev) // 10] * 1e3:.0f})")


def main_qkv(args):
    """RMSNorm + QKV + RoPE as one engine job (q/k head dims pair-interleaved)."""
    from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import pack_b_bfp16ebs8
    from shared.infra.cache import KernelCache, Profiler

    m, emb, nh, nkv, hd, tile_m, herd = 256, 960, 15, 5, 64, 32, 4
    kv = nkv * hd
    n = emb + 2 * kv
    tn, tk1 = args.tile_n, args.tk1
    tag = f"qkv_n{tn}_k{tk1}"
    cache = KernelCache(str(_HERE / "build" / f"gemm_engine_{tag}"), verbose=False, profiler=Profiler(enabled=True))
    sfx, obj = "_eng", "mm_engine.o"
    compile_mm_engine(tile_m, tn, tk1, sfx, obj, rms_k=emb)
    mod = build_gemm_engine(m, [Job("x", "wqkv", "qkv", emb, n, rms=True, rope="rope")],
                            tile_m, tn, tk1, tn * herd, herd, herd, sfx, obj, arg_order=["x", "wqkv", "rope", "qkv"])
    backend = {"verbose": False, "omit_while_true_loop": False, "output_format": "elf",
               "instance_name": "gemm_engine", "debug_ir": args.debug_ir}
    cache.compile_and_cache("qkv", mod, backend)

    rng = np.random.default_rng(0)
    f32 = np.float32

    def bf(x):
        return np.asarray(x, f32).astype(bfloat16)

    x = bf(rng.standard_normal((m, emb)))
    w = bf(rng.standard_normal((emb, n)) / np.sqrt(emb))
    nw = bf(1.0 + 0.1 * rng.standard_normal(emb))
    pos = np.minimum(np.arange(m), 240)
    inv = 1.0 / (10000.0 ** (np.arange(0, hd, 2) / hd))
    ang = np.outer(pos, inv)
    lut = bf(np.concatenate([np.cos(ang), np.sin(ang)], axis=1))
    perm = qkv_col_perm(nh, nkv, hd)
    wp = bf(nw.astype(f32)[:, None] * w.astype(f32))[:, perm]
    bufs = [x, pack_b_bfp16ebs8(wp, tn, tk1), bf(rope_table(lut.astype(f32), nh, nkv, hd, kv)),
            np.zeros((m, n), bfloat16)]

    def rope(a, heads):
        a = a.reshape(m, heads, hd)
        c, s_ = lut.astype(f32)[:, None, : hd // 2], lut.astype(f32)[:, None, hd // 2:]
        a1, a2 = a[..., : hd // 2], a[..., hd // 2:]
        return np.concatenate([a1 * c - a2 * s_, a2 * c + a1 * s_], axis=-1).reshape(m, -1)

    xf = x.astype(f32)
    ref = (xf / np.sqrt(np.mean(xf * xf, axis=1, keepdims=True) + 1e-5) * nw.astype(f32)) @ w.astype(f32)
    ref = np.concatenate([rope(ref[:, :emb], nh), rope(ref[:, emb:emb + kv], nkv), ref[:, emb + kv:]], axis=1)
    ref = ref[:, perm]

    def run():
        return cache.load_and_run("qkv", backend, *bufs, output_indices=[3], bo_key="qkv")

    out = np.asarray(run()[3], f32).reshape(m, n)
    from reconfig_probe import ctrl_kb
    print(f"  control code {ctrl_kb(cache.cache_dir):.1f} KB")
    for nm, sl in (("q", slice(0, emb)), ("k", slice(emb, emb + kv)), ("v", slice(emb + kv, n))):
        print(f"  {nm}: cosine {_cos(out[:, sl], ref[:, sl]):.6f}")
    cache.profiler.kernel_breakdowns.clear()
    for _ in range(args.iters):
        run()
    dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["qkv"])
    print(f"{tag}: device median {dev[len(dev) // 2] * 1e3:.0f} us (min {dev[0] * 1e3:.0f}, "
          f"p10 {dev[len(dev) // 10] * 1e3:.0f})")


if __name__ == "__main__":
    main()
