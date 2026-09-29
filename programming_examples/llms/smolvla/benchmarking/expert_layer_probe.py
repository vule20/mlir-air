# SPDX-License-Identifier: MIT
"""Feasibility probe: one SmolVLA action-expert layer as ONE NPU dispatch.

Same structure as the backbone's whole-layer ELF (backbone_npu.py --fused-layer
--offn-engine): rms+QKV+RoPE (4 launches) + masked FlashAttention (1) + the
gemm_engine O+FFN (1), stitched into one func, at the expert's shapes:
M = 50 action tokens (padded to 64), hidden 720 zero-padded to 960 (15 Q heads x
64 = 960 already), MLP 2048 zero-padded to 2240 (the engine's 320-column grain).
RMSNorm over the padded hidden: rms+QKV's norm divides by 960, so its weight
carries sqrt(720/960); the engine's divides by RMS_K = 720.

Attention reads separate K/V buffers of LK rows: LK=256 is a cross-attention
layer (241 prefix keys), LK=320 a self-attention layer (241 prefix + 50 action
keys). In a real self layer the action rows' K/V come from this layer's QKV
GEMM; here K/V are inputs for both, so the probe times the kernels and checks
them against an fp32 reference, not the full data flow.

CPU reference point: the expert's 10 denoise steps take ~153 ms in the pipeline
(8 bound threads, K/V memo), i.e. ~0.96 ms per layer-step.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import backbone_npu as bn  # noqa: F401  (sys.path setup)
from layer_fused import _privates, _signature_types
from shared.infra.stitching import FuncArg, KernelSlice, stitch_elf

M_REAL, M = 50, 64
E_REAL, E = 720, 960
H_REAL, H = 2048, 2240
NH, NKV, HD = 15, 5, 64
KV = NKV * HD
PREFIX = 241

# Combined args of the probe layer.
X, ANORM, NORMED, WQKV, QKV, ROPE_Q, Q_R, ROPE_K, K_R, KF, VF, MASK, ATTN, WO, RES1, WGU, SW, WDN, OUT = range(19)
RGR_MAP = {i: i for i in range(9)}
FA_MAP = {0: Q_R, 1: KF, 2: VF, 3: MASK, 4: ATTN}
ENG_ORDER = ["attn", "wo", "x", "res1", "wgu", "sw", "wdn", "out"]
ENG_MAP = {0: ATTN, 1: WO, 2: X, 3: RES1, 4: WGU, 5: SW, 6: WDN, 7: OUT}
STATIC = {ANORM, WQKV, ROPE_Q, ROPE_K, KF, VF, MASK, WO, WGU, WDN}
INTER = {NORMED, QKV, Q_R, K_R, ATTN, RES1, SW, OUT}
BACKEND = {"verbose": False, "omit_while_true_loop": False, "output_format": "elf",
           "instance_name": "layer", "runtime_loop_tiling_sizes": []}


def build(lk, fa_his, fa_qb, only=None, eng_tm=32):
    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import build_module
    from gemm_engine import Job, build_gemm_engine, compile_mm_engine
    from rms_gemms_rope_fused_qkv import build_rms_gemms_rope_module_fused_qkv
    import shared.infra.external_kernels as ek

    rgr = build_rms_gemms_rope_module_fused_qkv(M, E, KV, NH, NKV, HD, herd_m=1, qkv_tile_n=80,
                                                b_stationary=True, bfp16=(80, 480, 160))
    flags = ek._PEANO_FLAGS
    ek._PEANO_FLAGS = ["-Os" if f == "-O2" else f for f in flags]
    try:
        ek.compile_attn_npu2(head_dim=HD, bfp16=True, force=True)
    finally:
        ek._PEANO_FLAGS = flags
    fa = build_module(lk=lk, lkp=HD, lq=M, lqp=M, dk=HD, dv=HD, num_q_tiles=M // HD,
                      num_cascade_stages=lk // HD, num_heads=NH, num_kv_heads=NKV, num_heads_per_unroll=1,
                      causal=False, attn_mask=True, heads_in_segment=fa_his, q_bcast=fa_qb)
    compile_mm_engine(eng_tm, 80, 160, f"_xeng{eng_tm}", f"mm_xengine{eng_tm}.o", rms_k=E_REAL)
    eng = build_gemm_engine(
        M,
        [Job("attn", "wo", "res1", E, E, residual="x"),
         Job("res1", "wgu", "sw", E, 2 * H, rms=True, swiglu=True),
         Job("sw", "wdn", "out", H, E, residual="res1")],
        eng_tm, 80, 160, 320, M // eng_tm, 4, f"_xeng{eng_tm}", f"mm_xengine{eng_tm}.o", arg_order=ENG_ORDER,
    )
    parts = [("rg", str(rgr), RGR_MAP, [2, 5]), ("at", str(fa), FA_MAP, [2, 1]), ("of", str(eng), ENG_MAP, [])]
    types = [None] * 19
    for _, ir, amap, _ in parts:
        for op, t in enumerate(_signature_types(ir)):
            if op in amap:
                c = amap[op]
                assert types[c] in (None, t), (c, types[c], t)
                types[c] = t
    assert None not in types, types
    if only:
        parts = [pt for pt in parts if pt[0] in only]
    used = {c for _, _, amap, _ in parts for c in amap.values()}
    slices = [KernelSlice(ir, p, amap, extern_syms=_privates(ir)) for p, ir, amap, _ in parts]
    module = stitch_elf("layer", [FuncArg(f"%arg{i}", t) for i, t in enumerate(types)], slices,
                        debug_dump_path="/tmp/expert_probe_parse_error.mlir",
                        allow_unreferenced_args=set(range(19)) - used)
    from air.ir import DenseI64ArrayAttr

    func = next(op for op in module.body.operations
                if op.operation.name == "func.func" and op.attributes["sym_name"].value == "layer")
    launches = [op for op in func.regions[0].blocks[0].operations if op.operation.name == "air.launch"]
    per_launch = [ts for _, ir, _, ts in parts for _ in range(ir.count("air.launch "))]
    assert len(launches) == len(per_launch), (len(launches), len(per_launch))
    with module.context:
        for op, ts in zip(launches, per_launch):
            if ts:
                op.attributes["air.shim_dma_tile_sizes"] = DenseI64ArrayAttr.get(ts)
    print(f"  expert layer LK={lk}: {len(launches)} launches")
    return module


def rope_lut(n):
    inv = 1.0 / (10000.0 ** (np.arange(0, HD, 2) / HD))
    ang = np.outer(np.arange(n), inv)
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)


def rope_ref(x, lut, heads):
    x = x.reshape(x.shape[0], heads, HD)
    c, s = lut[:, None, : HD // 2], lut[:, None, HD // 2:]
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).reshape(x.shape[0], -1)


def make_data(lk, rng):
    from gemm_engine import permute_gate_up
    from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import pack_b_bfp16ebs8

    f32 = np.float32
    bf = lambda a: np.asarray(a, f32).astype(bfloat16)  # noqa: E731

    def pad(a, shape):
        out = np.zeros(shape, f32)
        out[tuple(slice(0, s) for s in a.shape)] = a
        return out

    x = pad(rng.standard_normal((M_REAL, E_REAL)), (M, E))
    anorm = pad(1 + 0.1 * rng.standard_normal(E_REAL), (E,))
    fnorm = pad(1 + 0.1 * rng.standard_normal(E_REAL), (E,))
    wq = pad(rng.standard_normal((E_REAL, NH * HD)) / np.sqrt(E_REAL), (E, NH * HD))
    wk = pad(rng.standard_normal((E_REAL, KV)) / np.sqrt(E_REAL), (E, KV))
    wv = pad(rng.standard_normal((E_REAL, KV)) / np.sqrt(E_REAL), (E, KV))
    wo = pad(rng.standard_normal((NH * HD, E_REAL)) / np.sqrt(NH * HD), (NH * HD, E))
    wg = pad(rng.standard_normal((E_REAL, H_REAL)) / np.sqrt(E_REAL), (E, H))
    wu = pad(rng.standard_normal((E_REAL, H_REAL)) / np.sqrt(E_REAL), (E, H))
    wd = pad(rng.standard_normal((H_REAL, E_REAL)) / np.sqrt(H_REAL), (H, E))
    kf = rng.standard_normal((lk, KV))
    vf = rng.standard_normal((lk, KV))
    valid = np.zeros((M, lk), bool)
    valid[:, :PREFIX] = True
    if lk > 256:
        valid[:, 256:256 + M_REAL] = True
    lut = rope_lut(M)
    wqkv = np.concatenate([wq, wk, wv], axis=1)
    w = dict(
        x=bf(x), anorm=bf(anorm * np.sqrt(E_REAL / E)), fnorm=bf(fnorm), wqkv=bf(wqkv), wo=bf(wo),
        wg=bf(wg), wu=bf(wu), wd=bf(wd), kf=bf(kf), vf=bf(vf), valid=valid, lut=bf(lut),
    )
    gu = permute_gate_up(bf(fnorm[:, None] * w["wg"].astype(f32)), bf(fnorm[:, None] * w["wu"].astype(f32)), 80, 320)
    z = lambda *s: np.zeros(s, bfloat16)  # noqa: E731
    args = [None] * 19
    args[X], args[ANORM], args[NORMED] = w["x"], w["anorm"], z(M, E)
    args[WQKV], args[QKV] = pack_b_bfp16ebs8(w["wqkv"], 80, 160), z(M, E + 2 * KV)
    args[ROPE_Q], args[Q_R] = np.repeat(w["lut"], NH, axis=0).flatten(), z(M, E)
    args[ROPE_K], args[K_R] = np.repeat(w["lut"], NKV, axis=0).flatten(), z(M, KV)
    args[KF], args[VF], args[MASK] = w["kf"], w["vf"], bn.additive_attn_mask(valid)
    args[ATTN], args[WO], args[RES1] = z(M, E), pack_b_bfp16ebs8(w["wo"], 80, 160), z(M, E)
    args[WGU], args[SW] = pack_b_bfp16ebs8(gu, 80, 160), z(M, H)
    args[WDN], args[OUT] = pack_b_bfp16ebs8(w["wd"], 80, 160), z(M, E)
    return args, w


def reference(w):
    f = lambda a: np.asarray(a, np.float32)  # noqa: E731
    x = f(w["x"])[:M_REAL, :E_REAL]

    def rms(a, g):
        return a / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-5) * g

    anorm = f(w["anorm"])[:E_REAL] / np.sqrt(E_REAL / E)
    qkv = rms(x, anorm) @ f(w["wqkv"])[:E_REAL]
    lut = f(w["lut"])[:M_REAL]
    q = rope_ref(qkv[:, :NH * HD], lut, NH).reshape(M_REAL, NH, HD)
    k, v = f(w["kf"]).reshape(-1, NKV, HD), f(w["vf"]).reshape(-1, NKV, HD)
    att = np.empty((M_REAL, NH, HD), np.float32)
    for h in range(NH):
        s = q[:, h] @ k[:, h // 3].T / np.sqrt(HD)
        s = np.where(w["valid"][:M_REAL], s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        att[:, h] = (p / p.sum(-1, keepdims=True)) @ v[:, h // 3]
    res1 = att.reshape(M_REAL, -1) @ f(w["wo"])[:, :E_REAL] + x
    n2 = rms(res1, f(w["fnorm"])[:E_REAL])
    g, u = n2 @ f(w["wg"])[:E_REAL, :H_REAL], n2 @ f(w["wu"])[:E_REAL, :H_REAL]
    sw = g / (1 + np.exp(-g)) * u
    return sw @ f(w["wd"])[:H_REAL, :E_REAL] + res1


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lk", type=int, nargs="+", default=[256, 320])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--fa-his", type=int, default=5)
    ap.add_argument("--no-qb", action="store_true")
    ap.add_argument("--eng-tile-m", type=int, default=32, help="engine tile_m (herd_m = 64 / tile_m)")
    ap.add_argument("--parts", default="rg,at,of", help="subset of rg (rms+QKV+RoPE), at (FA), of (O+FFN engine)")
    args = ap.parse_args()
    from shared.infra.cache import KernelCache, Profiler

    for lk in args.lk:
        only = args.parts.split(",")
        tag = f"lk{lk}_his{args.fa_his}{'' if args.no_qb else '_qb'}_{'-'.join(only)}_etm{args.eng_tile_m}"
        cache = KernelCache(str(Path(__file__).resolve().parent / "build" / f"expert_probe_{tag}"),
                            verbose=False, profiler=Profiler(enabled=True))
        cache.compile_and_cache("layer", build(lk, args.fa_his, not args.no_qb, only, args.eng_tile_m), BACKEND)
        bufs, w = make_data(lk, np.random.default_rng(0))
        run = lambda: cache.load_and_run(  # noqa: E731
            "layer", BACKEND, *bufs, output_indices=[OUT], static_input_indices=STATIC,
            intermediate_indices=INTER, bo_key="xl")
        out = np.asarray(run()[OUT], np.float32).reshape(M, E)
        if len(only) == 3:
            ref = reference(w)
            c = cos(out[:M_REAL, :E_REAL], ref)
            pad_max = float(np.abs(out[:M_REAL, E_REAL:]).max())
        else:
            c = pad_max = float("nan")
        cache.profiler.kernel_breakdowns.clear()
        for _ in range(args.reps):
            run()
        ks = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["layer"])
        n = len(ks)
        print(f"LK={lk}: out cosine vs fp32 ref {c:.6f} (pad cols max |x| {pad_max:.3g}); device per layer "
              f"min {ks[0] * 1e3:.0f} p10 {ks[n // 10] * 1e3:.0f} median {ks[n // 2] * 1e3:.0f} "
              f"p90 {ks[9 * n // 10] * 1e3:.0f} us  (CPU ~960 us per layer-step)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
