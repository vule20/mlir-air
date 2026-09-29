# SPDX-License-Identifier: MIT
"""How much of a launch's fixed cost does more work per launch amortize?

The O+FFN engine with its job list repeated for L independent layers (distinct
tensors per layer) in ONE launch. Core programs are loaded once per launch, but
the herd code and the shim puts are unrolled per job, so the control code still
grows with L. Reports control code and device time per L.

Every output drain is armed at launch start and a shim channel queues 4 tasks,
so more than 4 jobs with their own C tensors hang. --stack-c puts every job's C
in one tensor, drained by one task per channel (needs one C width: --jobs o).
--arena puts every activation in one tile-major arena (gemm_engine.ArenaLayout):
any job mix drains as one task per channel.
"""
import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

import backbone_npu as bn  # noqa: F401  (sys.path setup)
from gemm_engine import Job, _HERE, arena_layout, build_gemm_engine, compile_mm_engine, permute_gate_up


def _cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--jobs", default="o,gu,dn", help="per-layer job subset of o,gu,dn")
    ap.add_argument("--stack-c", action="store_true", help="all jobs' C in one stacked tensor (one drain)")
    ap.add_argument("--arena", action="store_true", help="all activations in one tile-major arena")
    ap.add_argument("--tag", default="", help="cache dir suffix (e.g. per compiler build)")
    ap.add_argument("--verbose", action="store_true", help="show the compiler's output")
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
            raw["wg" + s], raw["wu" + s] = wg, wu
            raw["wdn" + s] = bf(rng.standard_normal((hid, emb)) / 51)
            data.update({
                "attn" + s: bf(rng.standard_normal((m, emb))),
                "wo" + s: pack_b_bfp16ebs8(raw["wo" + s], tn, tk1),
                "x" + s: bf(rng.standard_normal((m, emb))),
                "res1" + s: bf(rng.standard_normal((m, emb))),
                "wgu" + s: pack_b_bfp16ebs8(permute_gate_up(wg, wu, tn, l2_n), tn, tk1),
                "sw" + s: bf(rng.standard_normal((m, hid))),
                "wdn" + s: pack_b_bfp16ebs8(raw["wdn" + s], tn, tk1),
                "out" + s: np.zeros((m, emb), bfloat16)})
        lay = arena_layout(m, jobs, tile_m, herd, l2_n) if args.arena else None
        if args.arena:
            order = [j.b for j in jobs] + ["act"]
            data["act"] = lay.empty()
            for name in lay.base:
                lay.pack(data["act"], name, data[name])
        else:
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
        tag += ("_sc" if args.stack_c else "") + ("_ar" if args.arena else "") + args.tag
        cache = KernelCache(str(_HERE / "build" / f"engine_rep_L{n_layers}{tag}"), verbose=False,
                            profiler=Profiler(enabled=True))
        backend = {"verbose": args.verbose, "omit_while_true_loop": False, "output_format": "elf",
                   "instance_name": "gemm_engine"}
        module = build_gemm_engine(m, jobs, tile_m, tn, tk1, l2_n, herd, herd, sfx, obj, arg_order=order,
                                   stack_c="cstack" if args.stack_c else None,
                                   arena="act" if args.arena else None)
        try:
            cache.compile_and_cache("eng", module, backend)
        except Exception as e:  # noqa: BLE001
            print(f"L={n_layers}: compile failed: {str(e)[-4000:] if args.verbose else str(e).splitlines()[-1][:200]}")
            continue
        if args.compile_only:
            print(f"L={n_layers}: compiled, control code {ctrl_kb(cache.cache_dir):.1f} KB")
            continue
        if args.arena:
            outs = [order.index("act")]
        elif args.stack_c:
            outs = [order.index("cstack")]
        else:
            outs = [order.index(j.c) for j in jobs]
        run = lambda: cache.load_and_run("eng", backend, *bufs, output_indices=outs, bo_key="e")  # noqa: E731
        res = run()
        if args.stack_c and kinds == ["o"]:
            got = np.array(res[outs[0]]).reshape(n_layers, m, emb)
            cos = [_cos(got[i], f(data["attn" + str(i)]) @ f(raw["wo" + str(i)]) + f(data["x" + str(i)]))
                   for i in range(n_layers)]
            print(f"L={n_layers}: per-layer cosine vs f32 ref min {min(cos):.6f} max {max(cos):.6f}")
        if args.arena:
            # Each job checked against f32 math on the NPU's own inputs (the arena after the run).
            act = np.array(res[outs[0]]).reshape(lay.empty().shape)
            got = lambda n: f(lay.unpack(act, n))  # noqa: E731
            cos = {}
            for i in range(n_layers):
                s = str(i)
                if "o" in kinds:
                    ref = f(data["attn" + s]) @ f(raw["wo" + s]) + f(data["x" + s])
                    cos.setdefault("o", []).append(_cos(got("res1" + s), ref))
                if "gu" in kinds:
                    r1 = got("res1" + s) if "o" in kinds else f(data["res1" + s])
                    rstd = 1.0 / np.sqrt((r1 * r1).mean(axis=1, keepdims=True) + 1e-5)
                    g, u = rstd * (r1 @ f(raw["wg" + s])), rstd * (r1 @ f(raw["wu" + s]))
                    cos.setdefault("gu", []).append(_cos(got("sw" + s), g / (1 + np.exp(-g)) * u))
                if "dn" in kinds:
                    r1 = got("res1" + s) if "o" in kinds else f(data["res1" + s])
                    sw = got("sw" + s) if "gu" in kinds else f(data["sw" + s])
                    cos.setdefault("dn", []).append(_cos(got("out" + s), sw @ f(raw["wdn" + s]) + r1))
            print(f"L={n_layers}: cosine vs f32 ref per job (min over layers): "
                  + ", ".join(f"{k} {min(v):.6f}" for k, v in cos.items()))
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
