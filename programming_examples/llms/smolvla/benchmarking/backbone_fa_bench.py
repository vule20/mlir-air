#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Standalone NPU FlashAttention at the SmolVLA backbone's attention shape.

seq 256 (241 real), 15 query / 5 KV heads, head_dim 64, seq-first layout (the
layout rms_gemms_rope already writes). Non-causal attn_npu2_seqfirst, unmasked
or (--mask) with the additive attn_mask=True variant, checked against an fp32
reference with the backbone's float32-min masking semantics.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent))

SEQ, NH, NKV, HD = 256, 15, 5, 64


def reference(q, k, v, mask=None):
    """fp32 attention; `mask` (bool, True = attend) is applied as an additive
    float32-min term, like backbone_npu.masked_attention_reference, so a fully
    masked row averages all keys."""
    q = q.astype(np.float32).reshape(SEQ, NH, HD).transpose(1, 0, 2)
    k = k.astype(np.float32).reshape(SEQ, NKV, HD).transpose(1, 0, 2)
    v = v.astype(np.float32).reshape(SEQ, NKV, HD).transpose(1, 0, 2)
    out = np.empty((NH, SEQ, HD), np.float32)
    for h in range(NH):
        s = q[h] @ k[h // (NH // NKV)].T / np.sqrt(HD)
        if mask is not None:
            s = s + np.where(mask, 0.0, np.finfo(np.float32).min).astype(np.float32)
        s -= s.max(-1, keepdims=True)
        p = np.exp(s)
        out[h] = (p / p.sum(-1, keepdims=True)) @ v[h // (NH // NKV)]
    return out.transpose(1, 0, 2).reshape(SEQ, NH * HD)


def additive_mask(mask_bool):
    """bool (True = attend) -> the bf16 additive mask attn_mask=True expects:
    0 where kept, bf16 lowest (0xff7f) where dropped."""
    m = np.zeros(mask_bool.shape, np.uint16)
    m[~mask_bool] = 0xFF7F
    return m.view(bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--hpu", type=int, default=1, help="num_heads_per_unroll")
    ap.add_argument("--bfp16", type=int, default=1, help="BFP16-emulated matmul microkernel")
    ap.add_argument("--pingpong", default="all", help="omit_pingpong value ('' = keep double buffers)")
    ap.add_argument("--tiling", default="1,1", help="runtime_loop_tiling_sizes")
    ap.add_argument("--mask", default="", help="bool (241,241) .npy mask (True = attend); "
                    "enables the additive-mask FA variant")
    ap.add_argument("--heads", default="15,5", help="q_heads,kv_heads (scaling probes)")
    ap.add_argument("--lqp", type=int, default=256)
    ap.add_argument("--nq", type=int, default=4, help="num_q_tiles")
    ap.add_argument("--opt", default="-O2", help="Peano opt level for attn_npu2.o (e.g. -Os: smaller core programs)")
    args = ap.parse_args()
    tiling = [int(t) for t in args.tiling.split(",")]
    global NH, NKV
    NH, NKV = (int(h) for h in args.heads.split(","))

    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import build_module
    from shared.infra.cache import KernelCache, Profiler
    from shared.infra.external_kernels import compile_attn_npu2

    tag = (f"hpu{args.hpu}_bfp{args.bfp16}_pp{args.pingpong or 'on'}_t{'x'.join(map(str, tiling))}"
           + ("_mask" if args.mask else "") + ("" if (NH, NKV) == (15, 5) else f"_h{NH}x{NKV}")
           + ("" if args.opt == "-O2" else args.opt.replace("-", "_"))
           + ("" if (args.lqp, args.nq) == (256, 4) else f"_lqp{args.lqp}nq{args.nq}"))
    cache = KernelCache(str(_HERE / "build" / f"backbone_fa_{tag}"),
                        verbose=False, profiler=Profiler(enabled=True))
    mod = build_module(lk=SEQ, lkp=HD, lq=SEQ, lqp=args.lqp, dk=HD, dv=HD, num_q_tiles=args.nq,
                       num_cascade_stages=4, num_heads=NH, num_kv_heads=NKV, causal=False,
                       num_heads_per_unroll=args.hpu, attn_mask=bool(args.mask))
    import shared.infra.external_kernels as ek

    ek._PEANO_FLAGS = [args.opt if f == "-O2" else f for f in ek._PEANO_FLAGS]
    compile_attn_npu2(head_dim=HD, bfp16=bool(args.bfp16), force=True)
    backend = {"verbose": False, "omit_while_true_loop": False, "omit_pingpong": args.pingpong,
               "runtime_loop_tiling_sizes": tiling, "output_format": "elf",
               "instance_name": "attention_bf16"}
    cache.compile_and_cache("flash_attn", mod, backend)

    rng = np.random.default_rng(0)
    q = (rng.standard_normal((SEQ, NH * HD)) * 1.5).astype(bfloat16)
    k = (rng.standard_normal((SEQ, NKV * HD)) * 1.5).astype(bfloat16)
    v = rng.standard_normal((SEQ, NKV * HD)).astype(bfloat16)
    o = np.zeros((SEQ, NH * HD), bfloat16)
    mask = None
    fa_args = [q, k, v, o]
    if args.mask:
        m241 = np.load(args.mask).astype(bool)
        mask = np.zeros((SEQ, SEQ), bool)
        mask[: m241.shape[0], : m241.shape[1]] = m241
        fa_args = [q, k, v, additive_mask(mask), o]
    out_idx = len(fa_args) - 1

    def run():
        return cache.load_and_run("flash_attn", backend, *fa_args, output_indices=[out_idx],
                                  bo_key="fa")[out_idx].reshape(SEQ, NH * HD)

    out = np.asarray(run(), dtype=np.float32)
    np.save(_HERE / "build" / f"backbone_fa_{tag}_out.npy", out)
    ref = reference(q, k, v, mask)

    def cos(a, b):
        return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))

    print(f"cosine vs fp32 reference ({'masked' if args.mask else 'unmasked'}): all rows "
          f"{cos(out, ref):.6f}  max abs err {np.abs(out - ref).max():.4f}  nan {np.isnan(out).sum()}")
    if args.mask:
        valid = mask.any(1)
        rc = [cos(out[i], ref[i]) for i in range(SEQ)]
        print(f"  rows attending >=1 key ({valid.sum()}): cosine {cos(out[valid], ref[valid]):.6f}, "
              f"worst row {min(rc[i] for i in np.where(valid)[0]):.6f}; fully-masked rows "
              f"({(~valid).sum()}): cosine {cos(out[~valid], ref[~valid]):.6f}")

    cache.profiler.kernel_breakdowns.clear()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        run()
    wall = (time.perf_counter() - t0) / args.iters * 1e3
    e = cache.profiler.kernel_breakdowns["flash_attn"]
    dev = sorted(x["kernel_ms"] for x in e)
    print(f"flash_attn {tag}: device median {dev[len(dev)//2]:.3f} ms "
          f"(min {dev[0]:.3f}), wall {wall:.3f} ms/call over {args.iters}")


if __name__ == "__main__":
    main()
