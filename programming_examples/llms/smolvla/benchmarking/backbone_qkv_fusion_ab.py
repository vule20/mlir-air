# A/B: separate Q/K/V (+ separate Gate/Up) launches vs fused-into-one, same
# herd_m=4 (today's confirmed optimum), same tiles. Isolates the fusion
# question cleanly -- no RMSNorm/RoPE IR surgery needed.
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_PROG = _HERE.parent.parent.parent
for p in (str(_PROG), str(_HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.builders.gemm_builder import _build_gemm_module
from shared.infra.external_kernels import compile_gemm_mm
from shared.infra.stitching import FuncArg, KernelSlice, stitch_elf

M = 256
HIDDEN, INTER = 960, 2560
Q_DIM, KV_DIM = 960, 320
HERD_M, HERD_N, TILE_M, TILE_N, TILE_K_L1 = 4, 4, 32, 80, 32


def pad_up(n, block):
    return ((n + block - 1) // block) * block


def build(tag, gemms, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    base_args, slices, shapes = [], [], []
    for i, (name, k, n) in enumerate(gemms):
        n_pad = pad_up(n, TILE_N * HERD_N)
        sfx = f"_{tag}_{name}"
        out_name = f"mm{sfx}.o"
        compile_gemm_mm(tile_m=TILE_M, tile_n=TILE_N, tile_k_l1=TILE_K_L1, sym_suffix=sfx, out_name=out_name)
        ir = str(
            _build_gemm_module(
                M, k, n_pad, TILE_M, k, TILE_K_L1, TILE_N, HERD_M, HERD_N,
                external_bf16_out=True, sym_suffix=sfx, link_with_name=out_name,
            )
        )
        a0 = 3 * i
        base_args += [
            FuncArg(f"%arg{a0}", f"memref<{M}x{k}xbf16>"),
            FuncArg(f"%arg{a0+1}", f"memref<{k}x{n_pad}xbf16>"),
            FuncArg(f"%arg{a0+2}", f"memref<{M}x{n_pad}xbf16>"),
        ]
        slices.append(
            KernelSlice(
                ir, f"{tag}{name}", {0: a0, 1: a0 + 1, 2: a0 + 2},
                extern_syms={
                    "@matmul_bf16",
                    "@op_has_no_registered_library_name" + sfx,
                    "@zero_f32_mn" + sfx,
                    "@f32_to_bf16_mn" + sfx,
                },
            )
        )
        shapes.append((name, k, n, n_pad))
    mod = stitch_elf(f"qkvfuse_{tag}", base_args, slices)

    from air.backend.xrt import XRTBackend
    backend = XRTBackend(
        verbose=False, omit_while_true_loop=False, runtime_loop_tiling_sizes=[2, 2],
        stack_size=2048, output_format="elf", instance_name=f"qkvfuse_{tag}",
        target_device="npu2", n_perf_iters=0,
    )
    return backend, backend.compile(mod), shapes


def time_it(tag, gemms, warmup=20, iters=100):
    import filelock, pyxrt as xrt
    backend, artifact, shapes = build(tag, gemms, _HERE / "build" / f"qkvfuse_{tag}")
    rng = np.random.default_rng(0)
    args_np, sizes = [], []
    for name, k, n, n_pad in shapes:
        A = rng.integers(-4, 4, size=(M, k)).astype(bfloat16)
        B = np.zeros((k, n_pad), dtype=bfloat16)
        B[:, :n] = rng.integers(-4, 4, size=(k, n)).astype(bfloat16)
        C = np.zeros((M, n_pad), dtype=bfloat16)
        args_np += [A, B, C]
        sizes += [a.size * a.itemsize for a in (A, B, C)]

    with filelock.FileLock("/tmp/npu.lock"):
        backend.load(artifact)
        bos = [xrt.ext.bo(backend.device, s) for s in sizes]
        bo_maps = [np.frombuffer(b.map(), dtype=np.uint8) for b in bos]

        def write_inputs():
            for i, a in enumerate(args_np):
                if i % 3 == 2:
                    continue
                src = np.ascontiguousarray(a.view(np.int16))
                bo_maps[i][: sizes[i]] = np.frombuffer(src.tobytes(), dtype=np.uint8)
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        def launch():
            run = xrt.run(backend.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            run.start()
            run.wait2()

        def read_all():
            for i in range(2, len(bos), 3):
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

        us = []
        for it in range(warmup + iters):
            write_inputs()
            t0 = time.perf_counter()
            launch()
            t1 = time.perf_counter()
            read_all()
            if it >= warmup:
                us.append((t1 - t0) * 1e6)
        backend.unload()
    us.sort()
    return sum(us) / len(us), us[0]


def main():
    UNFUSED_QKV = [("q", HIDDEN, Q_DIM), ("k", HIDDEN, KV_DIM), ("v", HIDDEN, KV_DIM)]
    FUSED_QKV = [("qkv", HIDDEN, Q_DIM + 2 * KV_DIM)]
    UNFUSED_GU = [("gate", HIDDEN, INTER), ("up", HIDDEN, INTER)]
    FUSED_GU = [("gu", HIDDEN, 2 * INTER)]

    for tag, gemms in [
        ("unfused_qkv", UNFUSED_QKV), ("fused_qkv", FUSED_QKV),
        ("unfused_gu", UNFUSED_GU), ("fused_gu", FUSED_GU),
    ]:
        avg, mn = time_it(tag, gemms)
        print(f"{tag:14s} avg={avg:8.1f}us  min={mn:8.1f}us  ({len(gemms)} launch{'es' if len(gemms)>1 else ''})")


if __name__ == "__main__":
    main()
