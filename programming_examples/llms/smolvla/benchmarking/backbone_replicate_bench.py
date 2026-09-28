#!/usr/bin/env python3
"""Per-launch fixed device cost of the backbone's GEMM shapes, by replication --
same method as replicate_bench2.py (vision) and profile.md §2.2:
    cost_per_launch = (t_hi - t_lo) / (hi - lo)      fixed ELF overhead cancels
Answers: how much of backbone_gemm_fused_bench.py's 2.09x-slower-than-CPU result
is fixed per-launch device cost (BD reprogram/lock) vs raw compute at M=241/256.

usage: backbone_replicate_bench.py {qkv,o,gu,dp} [--copies 2 4] [--iters 30]
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_PROG = _HERE.parent.parent.parent
for p in (str(_PROG), str(_HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("kernel", choices=["qkv", "o", "gu", "dp"])
ap.add_argument("--copies", type=int, nargs="+", default=[2, 4])
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--herd-m", type=int, default=None, help="override herd_m (default M//TILE_M0=8)")
ap.add_argument("--bstationary", action="store_true", help="tile_k_l2=K, b_stationary=True")
a = ap.parse_args()

WD = _HERE / "build" / "backbone_replicate"
WD.mkdir(parents=True, exist_ok=True)
os.chdir(WD)

from shared.infra.cache import KernelCache, Profiler
from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg
from shared.builders.gemm_builder import _build_gemm_module
from shared.infra.external_kernels import compile_gemm_mm

M = 256  # 241 padded
HIDDEN, INTER = 960, 2560
HEADS, KV_HEADS, HEAD_DIM = 15, 5, 64
Q_DIM, KV_DIM = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM

TILE_M0, TILE_N, HERD_N = 32, 96, 4


def pad_up(n, block):
    return ((n + block - 1) // block) * block


SPECS = {
    "qkv": (HIDDEN, Q_DIM + 2 * KV_DIM, 96, HIDDEN),
    "o": (Q_DIM, HIDDEN, 96, Q_DIM),
    "gu": (HIDDEN, 2 * INTER, 96, HIDDEN),
    "dp": (INTER, HIDDEN, 64, 256),
}
k, n, tk1, tk2 = SPECS[a.kernel]
n_pad = pad_up(n, TILE_N * HERD_N)
herd_m = a.herd_m if a.herd_m is not None else M // TILE_M0
assert M % (TILE_M0 * herd_m) == 0, (M, TILE_M0, herd_m)

bst_kwargs = {}
if a.bstationary:
    assert k % tk1 == 0, (k, tk1)
    tk2 = k
    bst_kwargs = {"b_stationary": True}

sfx = f"_rep_{a.kernel}_hm{herd_m}{'_bst' if a.bstationary else ''}"
out_name = f"mm{sfx}.o"
compile_gemm_mm(tile_m=TILE_M0, tile_n=TILE_N, tile_k_l1=tk1, sym_suffix=sfx, out_name=out_name)
ir = str(
    _build_gemm_module(
        M, k, n_pad, TILE_M0, tk2, tk1, TILE_N, herd_m, HERD_N,
        external_bf16_out=True, sym_suffix=sfx, link_with_name=out_name, **bst_kwargs,
    )
)
ext = {
    "@matmul_bf16",
    "@op_has_no_registered_library_name" + sfx,
    "@zero_f32_mn" + sfx,
    "@f32_to_bf16_mn" + sfx,
}
a_shape, b_shape, c_shape = (M, k), (k, n_pad), (M, n_pad)


def fa(shape):
    return f"memref<{'x'.join(map(str, shape))}xbf16>"


def build(nc, fname):
    args = [FuncArg("%arg0", fa(a_shape)), FuncArg("%arg1", fa(b_shape))] + [
        FuncArg(f"%arg{2 + i}", fa(c_shape)) for i in range(nc)
    ]
    sl = [
        KernelSlice(ir, f"c{i}", {0: 0, 1: 1, 2: 2 + i}, extern_syms=ext, private_from=(i == 0))
        for i in range(nc)
    ]
    return stitch_elf(fname, args, sl, debug_dump_path=f"/tmp/dbg_bbrep_{a.kernel}.mlir")


from air.backend.xrt import XRTBackend

variant = f"hm{herd_m}{'_bst' if a.bstationary else ''}"
cache = KernelCache(str(WD / f"cache_{a.kernel}_{variant}"), verbose=False, profiler=Profiler(enabled=True))
rng = np.random.default_rng(0)
A = (rng.standard_normal(a_shape) * 0.5).astype(bfloat16).reshape(-1)
Bw = (rng.standard_normal(b_shape) * 0.05).astype(bfloat16).reshape(-1)

res = {"kernel": a.kernel, "M": M, "K": k, "N": n, "variant": variant, "times_us": {}}
for nc in a.copies:
    name = f"bbrep_{a.kernel}_{variant}_x{nc}"
    kw = {
        "verbose": False,
        "omit_while_true_loop": False,
        "runtime_loop_tiling_sizes": [2, 2],
        "stack_size": 2048,
        "output_format": "elf",
        "target_device": "npu2",
        "n_perf_iters": 0,
        "instance_name": name,
    }
    t0 = time.time()
    cache.compile_and_cache(name, build(nc, name), kw)
    ct = time.time() - t0
    outs = [np.zeros(int(np.prod(c_shape)), bfloat16) for _ in range(nc)]
    idx = list(range(2, 2 + nc))
    for _ in range(a.warmup + a.iters):
        r = cache.load_and_run(
            name, kw, A, Bw, *outs, output_indices=[idx[0], idx[-1]],
            static_input_indices={1}, intermediate_indices=set(idx), bo_key=name,
        )
    ents = cache.profiler.kernel_breakdowns[name][a.warmup :]
    ts = [e["kernel_ms"] * 1e3 for e in ents]
    res["times_us"][nc] = {
        "mean": statistics.mean(ts), "median": statistics.median(ts), "min": min(ts), "compile_s": ct,
    }
    print(
        f"{a.kernel} x{nc}: median {statistics.median(ts):.1f} us  mean {statistics.mean(ts):.1f}"
        f"  min {min(ts):.1f}  (compile {ct:.0f}s)",
        flush=True,
    )

lo, hi = min(a.copies), max(a.copies)
tl, th = res["times_us"][lo]["median"], res["times_us"][hi]["median"]
res["per_launch_us"] = (th - tl) / (hi - lo)
res["fixed_elf_overhead_us"] = tl - lo * res["per_launch_us"]
flops = 2 * M * k * n
compute_us_per_launch = flops / (2.86e12) * 1e6  # CPU's own weighted-avg throughput, as a compute-time yardstick
print(
    f"RESULT {a.kernel}: per-launch {res['per_launch_us']:.1f} us, "
    f"fixed ELF overhead {res['fixed_elf_overhead_us']:.1f} us  "
    f"(real-shape compute at CPU's 2.86 TFLOP/s would take ~{compute_us_per_launch:.1f} us)",
    flush=True,
)
json.dump(res, open(WD / f"replicate_{a.kernel}.json", "w"), indent=1)
