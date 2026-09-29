# SPDX-License-Identifier: MIT
"""How much of a launch's fixed cost does more work per launch amortize?

The O+FFN engine (3 jobs) with its job list repeated for L independent layers
(distinct tensors per layer) in ONE launch. Core programs are loaded once per
launch, but the herd code and the shim puts are unrolled per job, so the control
code still grows with L. Reports control code and device time per L.
"""
import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

import backbone_npu as bn  # noqa: F401  (sys.path setup)
from gemm_engine import Job, _HERE, build_gemm_engine, compile_mm_engine, permute_gate_up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import pack_b_bfp16ebs8
    from reconfig_probe import ctrl_kb
    from shared.infra.cache import KernelCache, Profiler

    m, emb, hid, tile_m, tn, tk1, herd = 256, 960, 2560, 32, 80, 160, 4
    l2_n = tn * herd
    sfx, obj = "_eng", "mm_engine.o"
    compile_mm_engine(tile_m, tn, tk1, sfx, obj, rms_k=emb)
    rng = np.random.default_rng(0)
    bf = lambda a: np.asarray(a, np.float32).astype(bfloat16)  # noqa: E731
    for n_layers in args.layers:
        jobs, order, bufs = [], [], []
        for i in range(n_layers):
            s = str(i)
            jobs += [Job("attn" + s, "wo" + s, "res1" + s, emb, emb, residual="x" + s),
                     Job("res1" + s, "wgu" + s, "sw" + s, emb, 2 * hid, rms=True, swiglu=True),
                     Job("sw" + s, "wdn" + s, "out" + s, hid, emb, residual="res1" + s)]
            order += [n + s for n in ("attn", "wo", "x", "res1", "wgu", "sw", "wdn", "out")]
            wg, wu = bf(rng.standard_normal((emb, hid)) / 31), bf(rng.standard_normal((emb, hid)) / 31)
            bufs += [bf(rng.standard_normal((m, emb))), pack_b_bfp16ebs8(bf(rng.standard_normal((emb, emb)) / 31), tn, tk1),
                     bf(rng.standard_normal((m, emb))), np.zeros((m, emb), bfloat16),
                     pack_b_bfp16ebs8(permute_gate_up(wg, wu, tn, l2_n), tn, tk1), np.zeros((m, hid), bfloat16),
                     pack_b_bfp16ebs8(bf(rng.standard_normal((hid, emb)) / 51), tn, tk1), np.zeros((m, emb), bfloat16)]
        cache = KernelCache(str(_HERE / "build" / f"engine_rep_L{n_layers}"), verbose=False,
                            profiler=Profiler(enabled=True))
        backend = {"verbose": False, "omit_while_true_loop": False, "output_format": "elf",
                   "instance_name": "gemm_engine"}
        try:
            cache.compile_and_cache("eng", build_gemm_engine(m, jobs, tile_m, tn, tk1, l2_n, herd, herd, sfx, obj,
                                                             arg_order=order), backend)
        except Exception as e:  # noqa: BLE001
            print(f"L={n_layers}: compile failed: {str(e).splitlines()[-1][:200]}")
            continue
        outs = [8 * i + 7 for i in range(n_layers)]
        run = lambda: cache.load_and_run("eng", backend, *bufs, output_indices=outs, bo_key="e")  # noqa: E731
        run()
        cache.profiler.kernel_breakdowns.clear()
        for _ in range(args.iters):
            run()
        dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["eng"])
        med = dev[len(dev) // 2] * 1e3
        print(f"L={n_layers}: control code {ctrl_kb(cache.cache_dir):.1f} KB, device median {med:.0f} us "
              f"= {med / n_layers:.0f} us per layer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
