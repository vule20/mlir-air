#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""bf16 vs bfp16ebs8 weights for the SmolVLA backbone GEMMs at M=256.

At M=256 each weight element is used by only 256 rows, so the backbone GEMMs
should be bound by streaming the weights, the regime where bfp16 (9 bytes per 8
elements instead of 16) can pay for its unpacking. Standalone single-launch
ELFs, weights static, device median over --iters calls.

  bf16 : the production config (registry-driven or full-K B-stationary drain GEMM)
  bfp16: matrix_multiplication/bf16_x_bfp16 (A narrowed to bfp16 on the fly,
         f32 accumulator), --tk2 / --tk1 K tiling
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE.parent.parent), str(_HERE.parent.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

# name -> (K, N, bf16 tile_n, bf16 B-stationary)
SHAPES = {
    "qkv": (960, 1600, 80, True),
    "o": (960, 960, 80, False),
    "gu": (960, 5120, 128, True),
    "dn": (2560, 960, 80, False),
}
M, TILE_M, HERD = 256, 32, 4


def build_bf16(k, n, tile_n, bst):
    from shared.builders.gemm_builder import _build_gemm_module, gemm_registry_config
    from shared.infra.external_kernels import compile_gemm_mm

    if bst:
        compile_gemm_mm(tile_m=TILE_M, tile_n=tile_n, tile_k_l1=32, sym_suffix="_bb", out_name="mm_bb.o")
        return _build_gemm_module(M, k, n, TILE_M, k, 32, tile_n, HERD, HERD, external_bf16_out=True,
                                  sym_suffix="_bb", link_with_name="mm_bb.o", b_stationary=True)
    spec = gemm_registry_config(M, k, n, "bf16", "high")
    assert spec["tile_n"] == tile_n, spec
    kw = dict(spec["build_kwargs"], sym_suffix="_bb", link_with_name="mm_bb.o")
    compile_gemm_mm(tile_m=spec["tile_m"], tile_n=spec["tile_n"], tile_k_l1=spec["tile_k_l1"],
                    sym_suffix="_bb", out_name="mm_bb.o")
    return _build_gemm_module(M, k, n, spec["tile_m"], spec["tile_k_l2"], spec["tile_k_l1"], spec["tile_n"],
                              HERD, HERD, **kw)


def build_bfp16(k, n, tile_n, tk2, tk1, opt):
    from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import KERNEL_OBJ_NAME, build_module
    from shared.infra.external_kernels import _PROJ_ROOT, _get_aie_include_dir, _get_peano_clang

    subprocess.run(
        [_get_peano_clang(), opt, "-std=c++20", "--target=aie2p-none-unknown-elf", "-DNDEBUG",
         "-D__AIE_API_AIE_ADF_HPP__", "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
         f"-I{_get_aie_include_dir()}", f"-DDIM_M={TILE_M}", f"-DDIM_N={tile_n}", f"-DDIM_K={tk1}",
         "-c", str(_PROJ_ROOT / "matrix_multiplication" / "bf16_x_bfp16" / "mm_bf16_x_bfp16.cc"),
         "-o", KERNEL_OBJ_NAME],
        check=True,
    )
    return build_module(M, k, n, TILE_M, tk2, tk1, tile_n, HERD, HERD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=list(SHAPES), required=True)
    ap.add_argument("--variant", choices=["bf16", "bfp16", "bfp16l"], required=True,
                    help="bfp16l = gemm_bfp16.py (suffixed symbols, optional --swiglu)")
    ap.add_argument("--swiglu", action="store_true", help="bfp16l: SwiGLU drain, output n/2 wide")
    ap.add_argument("--tile-n", type=int, default=None, help="bfp16 tile_n (default: the bf16 one)")
    ap.add_argument("--tk2", type=int, default=None, help="bfp16 tile_k_l2 (default: K)")
    ap.add_argument("--tk1", type=int, default=64, help="bfp16 tile_k_l1")
    ap.add_argument("--opt", default="-O2")
    ap.add_argument("--tiling", default="2,2")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import pack_b_bfp16ebs8
    from shared.infra.cache import KernelCache, Profiler

    k, n, tn_bf16, bst = SHAPES[args.shape]
    tile_n = args.tile_n or tn_bf16
    tk2 = args.tk2 or k
    tiling = [int(t) for t in args.tiling.split(",")]
    tag = (f"{args.shape}_{args.variant}"
           + (f"_n{tile_n}_k{tk2}x{args.tk1}{args.opt.replace('-', '_')}" if args.variant != "bf16" else "")
           + ("_sw" if args.swiglu else "") + f"_t{'x'.join(map(str, tiling))}")
    cache = KernelCache(str(_HERE / "build" / f"bfp16bb_{tag}"), verbose=False, profiler=Profiler(enabled=True))
    if args.variant == "bf16":
        mod = build_bf16(k, n, tn_bf16, bst)
    elif args.variant == "bfp16":
        mod = build_bfp16(k, n, tile_n, tk2, args.tk1, args.opt)
    else:
        from gemm_bfp16 import build_gemm_bfp16, compile_mm_bfp16

        compile_mm_bfp16(TILE_M, tile_n, args.tk1, "_tst", "mm_bfp16_tst.o")
        mod = build_gemm_bfp16(M, k, n, TILE_M, tk2, args.tk1, tile_n, HERD, HERD, "_tst", "mm_bfp16_tst.o",
                               swiglu=args.swiglu)
    inst = "matmul_bf16" if args.variant == "bf16" else "matmul_bf16_x_bfp16"
    backend = {"verbose": False, "omit_while_true_loop": False, "output_format": "elf",
               "instance_name": inst, "runtime_loop_tiling_sizes": tiling}
    cache.compile_and_cache("gemm", mod, backend)

    rng = np.random.default_rng(0)
    a = (rng.standard_normal((M, k)) * 0.5).astype(bfloat16)
    w = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(bfloat16)
    wb = w
    if args.swiglu:
        from o_ffn_fused_gu import interleave_gate_up

        wb = interleave_gate_up(w[:, : n // 2], w[:, n // 2 :], tile_n // 2).astype(bfloat16)
    if args.variant != "bf16":
        wb = pack_b_bfp16ebs8(wb, tile_n, args.tk1)
    n_out = n // 2 if args.swiglu else n
    c = np.zeros((M, n_out), bfloat16)

    def run():
        return cache.load_and_run("gemm", backend, a, wb, c, output_indices=[2], static_input_indices={1},
                                  intermediate_indices={2}, bo_key="g")[2].reshape(M, n_out)

    out = np.asarray(run(), dtype=np.float32)
    ref = a.astype(np.float32) @ w.astype(np.float32)
    if args.swiglu:
        g, u = ref[:, : n // 2], ref[:, n // 2 :]
        ref = g / (1 + np.exp(-g)) * u
    cos = float(out.ravel() @ ref.ravel() / (np.linalg.norm(out) * np.linalg.norm(ref)))
    rel = float(np.abs(out - ref).max() / np.abs(ref).max())
    cache.profiler.kernel_breakdowns.clear()
    for _ in range(args.iters):
        run()
    dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["gemm"])
    wmb = np.asarray(wb).nbytes / 2**20
    q = [dev[int(len(dev) * f)] * 1e3 for f in (0.25, 0.75)]
    print(f"{tag}: device median {dev[len(dev)//2]*1e3:.0f} us (min {dev[0]*1e3:.0f}, "
          f"q1 {q[0]:.0f}, q3 {q[1]:.0f}), weights {wmb:.2f} MiB, "
          f"cosine {cos:.6f}, max err/max|ref| {rel:.4f}")


if __name__ == "__main__":
    main()
