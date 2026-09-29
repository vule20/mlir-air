# SPDX-License-Identifier: MIT
"""How much of a launch's fixed cost does more work per launch amortize?

The O+FFN engine with its job list repeated for L independent layers (distinct
tensors per layer) in ONE launch. Core programs are loaded once per launch, but
the herd code and the shim puts are unrolled per job, so the control code still
grows with L. Reports control code and device time per L.

Every output drain is armed at launch start and a shim channel queues 4 tasks,
so more than 4 jobs with their own C tensors hang. --stack-c puts every job's C
in one tensor, drained by one task per channel (needs one C width: --jobs o).
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
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--jobs", default="o,gu,dn", help="per-layer job subset of o,gu,dn")
    ap.add_argument("--stack-c", action="store_true", help="all jobs' C in one stacked tensor (one drain)")
    ap.add_argument("--tag", default="", help="cache dir suffix (e.g. per compiler build)")
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
    f = lambda a: np.asarray(a, np.float32)  # noqa: E731
    for n_layers in args.layers:
        jobs, data, raw = [], {}, {}
        kinds = args.jobs.split(",")
        for i in range(n_layers):
            s = str(i)
            job = {"o": Job("attn" + s, "wo" + s, "res1" + s, emb, emb, residual="x" + s),
                   "gu": Job("res1" + s, "wgu" + s, "sw" + s, emb, 2 * hid, rms=True, swiglu=True),
                   "dn": Job("sw" + s, "wdn" + s, "out" + s, hid, emb, residual="res1" + s)}
            jobs += [job[k] for k in kinds]
            wg, wu = bf(rng.standard_normal((emb, hid)) / 31), bf(rng.standard_normal((emb, hid)) / 31)
            raw["wo" + s] = bf(rng.standard_normal((emb, emb)) / 31)
            data.update({
                "attn" + s: bf(rng.standard_normal((m, emb))),
                "wo" + s: pack_b_bfp16ebs8(raw["wo" + s], tn, tk1),
                "x" + s: bf(rng.standard_normal((m, emb))),
                "res1" + s: bf(rng.standard_normal((m, emb))),
                "wgu" + s: pack_b_bfp16ebs8(permute_gate_up(wg, wu, tn, l2_n), tn, tk1),
                "sw" + s: bf(rng.standard_normal((m, hid))),
                "wdn" + s: pack_b_bfp16ebs8(bf(rng.standard_normal((hid, emb)) / 51), tn, tk1),
                "out" + s: np.zeros((m, emb), bfloat16)})
        order = []
        for j in jobs:
            for n in (j.a, j.b, j.residual, None if args.stack_c else j.c):
                if n and n not in order:
                    order.append(n)
        if args.stack_c:
            order.append("cstack")
            data["cstack"] = np.zeros((len(jobs) * m, jobs[0].n_out), bfloat16)
        bufs = [data[n] for n in order]
        tag = "" if args.jobs == "o,gu,dn" else "_" + args.jobs.replace(",", "")
        tag += ("_sc" if args.stack_c else "") + args.tag
        cache = KernelCache(str(_HERE / "build" / f"engine_rep_L{n_layers}{tag}"), verbose=False,
                            profiler=Profiler(enabled=True))
        backend = {"verbose": False, "omit_while_true_loop": False, "output_format": "elf",
                   "instance_name": "gemm_engine"}
        module = build_gemm_engine(m, jobs, tile_m, tn, tk1, l2_n, herd, herd, sfx, obj, arg_order=order,
                                   stack_c="cstack" if args.stack_c else None)
        try:
            cache.compile_and_cache("eng", module, backend)
        except Exception as e:  # noqa: BLE001
            print(f"L={n_layers}: compile failed: {str(e).splitlines()[-1][:200]}")
            continue
        if args.compile_only:
            print(f"L={n_layers}: compiled, control code {ctrl_kb(cache.cache_dir):.1f} KB")
            continue
        outs = [order.index("cstack")] if args.stack_c else [order.index(j.c) for j in jobs]
        run = lambda: cache.load_and_run("eng", backend, *bufs, output_indices=outs, bo_key="e")  # noqa: E731
        res = run()
        if args.stack_c and kinds == ["o"]:
            got = np.array(res[outs[0]]).reshape(n_layers, m, emb)
            cos = []
            for i in range(n_layers):
                s = str(i)
                ref = (f(data["attn" + s]) @ f(raw["wo" + s]) + f(data["x" + s])).ravel()
                g = f(got[i]).ravel()
                cos.append(float(g @ ref / (np.linalg.norm(g) * np.linalg.norm(ref) + 1e-30)))
            print(f"L={n_layers}: per-layer cosine vs f32 ref min {min(cos):.6f} max {max(cos):.6f}")
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
