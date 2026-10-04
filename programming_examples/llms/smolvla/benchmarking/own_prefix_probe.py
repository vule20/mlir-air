# SPDX-License-Identifier: MIT
"""Own-key scores over several passes of 64 keys (Job.own_passes): prefix keys read as bf16 from the
arena through the B channel, like the step's own keys. One score job, random data, vs numpy.

    python own_prefix_probe.py [--passes 4] [--run-only]

Output column of pass p, tile column c*80 + j*16 + kk = (head j of Q tile 0, key 64p + 16c + kk)."""
import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import backbone_npu as bn  # noqa: F401  (sys.path setup)
import expert_engine_probe as xp
from gemm_engine import Job, arena_layout, build_gemm_engine, compile_mm_engine, weights_layout

ap = argparse.ArgumentParser()
ap.add_argument("--passes", type=int, default=4)
ap.add_argument("--run-only", action="store_true")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--packed", type=int, default=0, help="timing only: N packed bfp16 output tiles, no own tiles (today's prefix path)")
ap.add_argument("--pv", action="store_true", help="also the P.V job over the same passes (Job(own='pv'))")
ap.add_argument("--lead", type=int, default=0, help="packed (host-packed bfp16, here all-masked zeros) tiles before the own tiles, as in the layer")
ap.add_argument("--proj", action="store_true", help="K/V come from projection jobs (cross layer): kvp{p} = [kc{p} @ Wk | vc{p} @ Wv]; needs --pv, --lead 0")
ap.add_argument("--self-layer", action="store_true", help="with --proj: a self-attention layer: K = RoPE(-p0)(kc) (constant block-diagonal matrix, pair-interleaved columns), V = vc (identity weights)")
ap.add_argument("--p0", type=int, default=137)
ap.add_argument("--masked", type=float, default=0.25, help="fraction of keys masked out")
args = ap.parse_args()
P = args.passes
M, L2N, TN, TK1, HERD, TILE_M, HPT, HD, QPG = xp.M, xp.L2N, xp.TN, xp.TK1, xp.HERD, xp.TILE_M, xp.HPT, xp.HD, 3

from air.backend.xrt import XRTCompileArtifact
from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import pack_b_bfp16ebs8
from shared.infra.cache import KernelCache, Profiler

compile_mm_engine(TILE_M, TN, TK1, xp.SFX, xp.OBJ, rms_k=xp.E_REAL)
bf = lambda a: np.asarray(a, np.float32).astype(bfloat16)  # noqa: E731
jobs = [] if args.proj else [Job("q", "wd", "y", L2N, L2N)]  # a normal job, so there are weights
if args.proj:
    assert args.pv and not args.lead and not args.packed and P >= 3
    for p in range(P):
        jobs += [Job(f"kc{p}", "wr" if args.self_layer else "wk", f"kvp{p}", L2N, L2N),
                 Job(f"vc{p}", "wid" if args.self_layer else "wv", f"kvp{p}", L2N, L2N, c_off=L2N)]
    jobs.append(Job("q", "kbs", "p", L2N, P * L2N, residual="mask", exp=True, own="s", kv="kvp", kv_off=0, own_passes=P))
    jobs.append(Job("p", "vb", "attn", P * L2N, 2 * L2N, div=True, own="pv", kv="kvp", kv_off=0, own_passes=P))
elif args.packed:
    jobs.append(Job("q", "kbs", "p", L2N, args.packed * L2N, residual="mask", exp=True))
    if args.pv:
        jobs.append(Job("p", "vb", "attn", args.packed * L2N, 2 * L2N, div=True))
    P = args.packed
else:
    LD = args.lead
    jobs.append(Job("q", "kbs", "p", L2N, (LD + P) * L2N, residual="mask", exp=True, own="s", kv="kvp", kv_off=0, own_passes=P))
    if args.pv:
        jobs.append(Job("p", "vb", "attn", (LD + P) * L2N, 2 * L2N, div=True, own="pv", kv="kvp", kv_off=0, own_passes=P))
lay = arena_layout(M, jobs, TILE_M, HERD, L2N)
wbase, wrows = weights_layout(jobs, TN, L2N)
cache = KernelCache(str(Path(__file__).resolve().parent / "build" / f"own_prefix_{"pk" if args.packed else "P"}{P}{"_pv" if args.pv else ""}{"_ld" + str(args.lead) if args.lead else ""}{"_proj" if args.proj else ""}{"self" if args.self_layer else ""}"), verbose=False,
                    profiler=Profiler(enabled=True))
backend = dict(xp.BACKEND)
elf = cache.cache_dir / "eng.elf"
if args.run_only and elf.exists():
    cache.artifacts["eng"] = XRTCompileArtifact(str(elf), "main:gemm_engine", None)
else:
    module = build_gemm_engine(M, jobs, TILE_M, TN, TK1, L2N, HERD, HERD, xp.SFX, xp.OBJ,
                               arg_order=["wts", "act"], arena="act", weights="wts", shim_at_launch=True)
    cache.compile_and_cache("eng", module, backend)

rng = np.random.default_rng(0)
from gemm_engine import rope_pair_perm as _rpp
rope_pair_perm_q = _rpp(HPT, HD)  # Q tile's 5 heads pair-interleaved like the K columns (scores are invariant)
Q = bf(0.3 * rng.standard_normal((M, L2N)))                       # Q tile 0: 5 heads x 64 dims
KV = [bf(0.3 * rng.standard_normal((M, 2 * L2N))) for _ in range(P)]  # per pass: K | V, [64 keys, 640]
if args.proj:
    KC = [bf(0.3 * rng.standard_normal((M, L2N))) for _ in range(P)]   # backbone K / V cache rows of pass p
    VC = [bf(0.3 * rng.standard_normal((M, L2N))) for _ in range(P)]
    WK, WV = [bf(rng.standard_normal((L2N, L2N)) / np.sqrt(L2N)) for _ in range(2)]
    KV = [np.concatenate([KC[p].astype(np.float32) @ WK.astype(np.float32),
                          VC[p].astype(np.float32) @ WV.astype(np.float32)], axis=1) for p in range(P)]
    if args.self_layer:
        import expert_capture as ec
        from gemm_engine import rope_pair_perm
        NKV_ = L2N // HD
        kperm = rope_pair_perm(NKV_, HD)
        # R[i, :] = RoPE(-p0) of unit vector i, so k_rot = k @ R; columns pair-interleaved like Q.
        R = ec.rope(np.eye(L2N, dtype=np.float32).reshape(L2N, NKV_, HD), np.full(L2N, -args.p0), 10000.0).reshape(L2N, L2N)
        WK, WV = bf(R[:, kperm]), bf(np.eye(L2N))
        # what the layer wants (reference basis: K unrotated-columns, Q unpermuted)
        KV = [np.concatenate([ec.rope(KC[p].astype(np.float32).reshape(M, NKV_, HD), np.full(M, -args.p0), 10000.0).reshape(M, L2N),
                              VC[p].astype(np.float32)], axis=1) for p in range(P)]
keep = rng.random((M, P * 64)) >= args.masked                     # [query, key]
keep[:, 0] = True
act = lay.empty()
lay.pack(act, "q", Q[:, rope_pair_perm_q] if args.self_layer else Q)
for p in range(0 if args.packed else P):
    if args.proj:
        lay.pack(act, f"kc{p}", KC[p])
        lay.pack(act, f"vc{p}", VC[p])
    else:
        lay.pack(act, "kvp" if P == 1 else f"kvp{p}", KV[p])
# mask in the output column order of each pass tile: c*80 + j*16 + kk <-> key 64p + 16c + kk
cols = np.arange(L2N)
c_, j_, kk_ = cols // (HPT * 16), (cols % (HPT * 16)) // 16, cols % 16
mask = np.concatenate([np.full((M, L2N), -1e30)] * args.lead + [np.where(keep[:, 64 * p + 16 * c_ + kk_], 0.0, -1e30) for p in range(P)], axis=1)
lay.pack(act, "mask", bf(mask))
wts = np.zeros((wrows, L2N // TK1, pack_b_bfp16ebs8(bf(np.zeros((L2N, TN))), TN, TK1).shape[-1]), np.uint8)
if args.proj:
    for nm, W in ((("wr", WK), ("wid", WV)) if args.self_layer else (("wk", WK), ("wv", WV))):
        rows = pack_b_bfp16ebs8(W, TN, TK1).reshape(-1, L2N // TK1, wts.shape[-1])
        wts[wbase[nm]:wbase[nm] + len(rows)] = rows
if args.packed:
    got = np.asarray(cache.load_and_run("eng", backend, wts, act, output_indices=[1], bo_key="op")[1])
    cache.profiler.kernel_breakdowns.clear()
    for _ in range(args.iters):
        cache.load_and_run("eng", backend, wts, act, output_indices=[1], bo_key="op")
    dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["eng"])
    print(f"packed {args.packed} tiles: device median {dev[len(dev) // 2] * 1e3:.0f} us")
    sys.exit(0)

got = np.asarray(cache.load_and_run("eng", backend, wts, act, output_indices=[1], bo_key="op")[1]).reshape(act.shape)
out = lay.unpack(got, "p").astype(np.float32)[:, args.lead * L2N:]  # [64, P*320]

ref = np.zeros((M, P * L2N), np.float32)
Qf = Q.astype(np.float32)
for p in range(P):
    K = KV[p].astype(np.float32)[:, :L2N]  # [64 keys, 5 groups x 64]
    for c in range(HERD):
        for j in range(HPT):
            g = j // QPG  # head_t = 0: heads 0..4 -> groups 0,0,0,1,1
            s = Qf[:, j * HD:(j + 1) * HD] @ K[16 * c:16 * c + 16, g * HD:(g + 1) * HD].T  # [64, 16]
            ref[:, p * L2N + c * (HPT * 16) + j * 16: p * L2N + c * (HPT * 16) + j * 16 + 16] = np.exp(s + mask[:, (args.lead + p) * L2N + c * (HPT * 16) + j * 16: (args.lead + p) * L2N + c * (HPT * 16) + j * 16 + 16])
cos = lambda a, b: float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
print(f"P={P}: scores vs numpy cosine {cos(out, ref):.6f}, max|d| {np.abs(out - ref).max():.4g} (max |ref| {ref.max():.3g}); "
      f"per pass: " + " ".join(f"{cos(out[:, p * L2N:(p + 1) * L2N], ref[:, p * L2N:(p + 1) * L2N]):.5f}" for p in range(P)))
if args.pv:
    att = lay.unpack(got, "attn").astype(np.float32)  # [64, 320]: head j, dim d = col j*64 + d
    num, den = np.zeros((M, L2N), np.float32), np.zeros((M, HPT), np.float32)
    for p in range(P):
        V = KV[p].astype(np.float32)[:, L2N:]  # [64 keys, 5 groups x 64]
        for c in range(HERD):
            for j in range(HPT):
                g = j // QPG
                col = p * L2N + c * (HPT * 16) + j * 16
                pk = ref[:, col:col + 16]  # [64, 16 keys]
                num[:, j * HD:(j + 1) * HD] += pk @ V[16 * c:16 * c + 16, g * HD:(g + 1) * HD]
                den[:, j] += pk.sum(axis=1)
    att_ref = num / np.repeat(den, HD, axis=1)
    bad = ~np.isfinite(att)
    print(f"  attn: {bad.sum()} non-finite of {att.size}; zeros {(att == 0).sum()}; rows with nan {np.where(bad.any(axis=1))[0][:8]}; "
          f"cols with nan {np.where(bad.any(axis=0))[0][:12]}; finite cosine {cos(np.where(bad, 0, att), np.where(bad, 0, att_ref)):.5f}")
    print("  finite rows:", np.where(~bad.any(axis=1))[0].tolist(), " nan per row-block of 16:", [int(bad[i:i + 16].sum()) for i in range(0, 64, 16)])
    fin = np.where(~bad.any(axis=0))[0]
    print("  finite columns:", fin.tolist()[:80], "... n =", len(fin))
    for nm in ("attn",):
        raw = lay.unpack(got, nm).astype(np.float32)
        print("  raw attn[0, 0:12]:", raw[0, :12], " ref:", att_ref[0, :12])
    print(f"P={P}: attention (P.V / P.1) vs numpy cosine {cos(att, att_ref):.6f}, max|d| {np.abs(att - att_ref).max():.4g} "
          f"(max |ref| {np.abs(att_ref).max():.3g})")
cache.profiler.kernel_breakdowns.clear()
for _ in range(args.iters):
    cache.load_and_run("eng", backend, wts, act, output_indices=[1], bo_key="op")
dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["eng"])
print(f"P={P}: device median {dev[len(dev) // 2] * 1e3:.0f} us")
