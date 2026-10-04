# SPDX-License-Identifier: MIT
"""Prefix launch + step launch: two engine ELFs, one shared K/V buffer.

The prefix engine turns the backbone's K/V rows (kc{p}, vc{p}, 64 keys per pass) into K|V tiles
kvp{p} (once per chunk); the step engine's own-key score and PV jobs read them through a third
argument, `kv`, whose buffer IS the prefix engine's arena buffer (kv_lay = the prefix layout).

    python prefix_engine_probe.py [--passes 4] [--run-only]
"""
import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import backbone_npu as bn  # noqa: F401  (sys.path setup)
import expert_engine_probe as xp
from gemm_engine import Job, arena_layout, build_gemm_engine, compile_mm_engine, weights_layout

ap = argparse.ArgumentParser()
ap.add_argument("--passes", type=int, default=4)
ap.add_argument("--run-only", action="store_true")
ap.add_argument("--layers", type=int, default=1, help="layers in the prefix launch (the step engine uses layer 0)")
ap.add_argument("--iters", type=int, default=60)
args = ap.parse_args()
P = args.passes
M, L2N, TN, TK1, HERD, TILE_M, HPT, HD, QPG = xp.M, xp.L2N, xp.TN, xp.TK1, xp.HERD, xp.TILE_M, xp.HPT, xp.HD, 3

from air.backend.xrt import XRTCompileArtifact
from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import pack_b_bfp16ebs8
from shared.infra.cache import KernelCache, Profiler

compile_mm_engine(TILE_M, TN, TK1, xp.SFX, xp.OBJ, rms_k=xp.E_REAL)
bf = lambda a: np.asarray(a, np.float32).astype(bfloat16)  # noqa: E731

# prefix engine: kvp{p} = [kc{p} @ Wk | vc{p} @ Wv]
NL = args.layers
jobs_pre = []
for l in range(NL):
    for p in range(P):
        jobs_pre += [Job(f"kc{l}_{p}", f"wk{l}", f"kvp{l}_{p}", L2N, L2N), Job(f"vc{l}_{p}", f"wv{l}", f"kvp{l}_{p}", L2N, L2N, c_off=L2N)]
lay_pre = arena_layout(M, jobs_pre, TILE_M, HERD, L2N, pad_tiles=2)
wbase_pre, wrows_pre = weights_layout(jobs_pre, TN, L2N)
# step engine: scores + P.V over the P passes, K|V read from the external kv arena
srcs = tuple((f"kvp0_{p}", 0) for p in range(P))
jobs_step = [Job("q", "wd", "y", L2N, L2N),
             Job("q", "kbs", "p", L2N, P * L2N, residual="mask", exp=True, own="s", kv="kvp", own_passes=P, kv_srcs=srcs),
             Job("p", "vb", "attn", P * L2N, 2 * L2N, div=True, own="pv", kv="kvp", own_passes=P, kv_srcs=srcs)]
lay = arena_layout(M, jobs_step, TILE_M, HERD, L2N, external=lay_pre.base)
wbase, wrows = weights_layout(jobs_step, TN, L2N)

cache = KernelCache(str(Path(__file__).resolve().parent / "build" / f"prefix_engine_P{P}_L{NL}"), verbose=False,
                    profiler=Profiler(enabled=True))
backend = dict(xp.BACKEND)
elf = {nm: cache.cache_dir / f"{nm}.elf" for nm in ("pre", "step")}
if args.run_only and all(e.exists() for e in elf.values()):
    for nm in elf:
        cache.artifacts[nm] = XRTCompileArtifact(str(elf[nm]), "main:gemm_engine", None)
else:
    cache.compile_and_cache("pre", build_gemm_engine(M, jobs_pre, TILE_M, TN, TK1, L2N, HERD, HERD, xp.SFX, xp.OBJ,
                            arg_order=["wts", "act"], arena="act", weights="wts", shim_at_launch=True), backend)
    cache.compile_and_cache("step", build_gemm_engine(M, jobs_step, TILE_M, TN, TK1, L2N, HERD, HERD, xp.SFX, xp.OBJ,
                            arg_order=["wts", "act", "kv"], arena="act", weights="wts", shim_at_launch=True,
                            kv_arena="kv", kv_lay=lay_pre), backend)

rng = np.random.default_rng(0)
Q = bf(0.3 * rng.standard_normal((M, L2N)))
KC = [bf(0.3 * rng.standard_normal((M, L2N))) for _ in range(P)]
VC = [bf(0.3 * rng.standard_normal((M, L2N))) for _ in range(P)]
WK, WV = [bf(rng.standard_normal((L2N, L2N)) / np.sqrt(L2N)) for _ in range(2)]
KV = [np.concatenate([KC[p].astype(np.float32) @ WK.astype(np.float32),
                      VC[p].astype(np.float32) @ WV.astype(np.float32)], axis=1) for p in range(P)]
keep = rng.random((M, P * 64)) >= 0.25
keep[:, 0] = True
cols = np.arange(L2N)
c_, j_, kk_ = cols // (HPT * 16), (cols % (HPT * 16)) // 16, cols % 16
mask = np.concatenate([np.where(keep[:, 64 * p + 16 * c_ + kk_], 0.0, -1e30) for p in range(P)], axis=1)

nbytes = pack_b_bfp16ebs8(bf(np.zeros((L2N, TN))), TN, TK1).shape[-1]
wts_pre = np.zeros((wrows_pre, L2N // TK1, nbytes), np.uint8)
for l in range(NL):
    for nm, W in ((f"wk{l}", WK), (f"wv{l}", WV)):
        rows = pack_b_bfp16ebs8(W, TN, TK1).reshape(-1, L2N // TK1, nbytes)
        wts_pre[wbase_pre[nm]:wbase_pre[nm] + len(rows)] = rows
act_pre = lay_pre.empty()
for l in range(NL):
    for p in range(P):
        lay_pre.pack(act_pre, f"kc{l}_{p}", KC[p])
        lay_pre.pack(act_pre, f"vc{l}_{p}", VC[p])
wts_step = np.zeros((wrows, L2N // TK1, nbytes), np.uint8)
act = lay.empty()
lay.pack(act, "q", Q)
lay.pack(act, "mask", bf(mask))

# 1. prefix launch; 2. step launch whose kv argument is the prefix launch's arena buffer.
cache.load_and_run("pre", backend, wts_pre, act_pre, output_indices=[1], bo_key="pre")
cache.load_and_run("step", backend, wts_step, act, act_pre, output_indices=[1], bo_key="step",
                   static_input_indices={0})
cache._cached_bos["step"][2] = cache._cached_bos["pre"][1]  # share the buffer object
got = np.asarray(cache.load_and_run("step", backend, wts_step, act, act_pre, output_indices=[1], bo_key="step",
                                    static_input_indices={0, 2})[1]).reshape(act.shape)

out = lay.unpack(got, "p").astype(np.float32)
att = lay.unpack(got, "attn").astype(np.float32)
Qf = Q.astype(np.float32)
ref = np.zeros((M, P * L2N), np.float32)
num, den = np.zeros((M, L2N), np.float32), np.zeros((M, HPT), np.float32)
for p in range(P):
    K, V = KV[p][:, :L2N], KV[p][:, L2N:]
    for c in range(HERD):
        for j in range(HPT):
            g = j // QPG
            col = p * L2N + c * (HPT * 16) + j * 16
            sc = Qf[:, j * HD:(j + 1) * HD] @ K[16 * c:16 * c + 16, g * HD:(g + 1) * HD].T
            ref[:, col:col + 16] = np.exp(sc + mask[:, col:col + 16])
            num[:, j * HD:(j + 1) * HD] += ref[:, col:col + 16] @ V[16 * c:16 * c + 16, g * HD:(g + 1) * HD]
            den[:, j] += ref[:, col:col + 16].sum(axis=1)
att_ref = num / np.repeat(den, HD, axis=1)
cos = lambda a, b: float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
print(f"P={P}: prefix launch -> step launch: scores cosine {cos(out, ref):.6f}; attention cosine {cos(att, att_ref):.6f} "
      f"(max|d| {np.abs(att - att_ref).max():.4g}, non-finite {int((~np.isfinite(att)).sum())})")
for nm, fn in (("prefix", lambda: cache.load_and_run("pre", backend, wts_pre, act_pre, output_indices=[], bo_key="pre",
                                                     static_input_indices={0}, intermediate_indices={1})),
               ("step", lambda: cache.load_and_run("step", backend, wts_step, act, act_pre, output_indices=[1], bo_key="step",
                                                   static_input_indices={0, 2}))):
    cache.profiler.kernel_breakdowns.clear()
    for _ in range(args.iters):
        fn()
    key = "pre" if nm == "prefix" else "step"
    dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns[key])
    print(f"{nm} launch: device median {dev[len(dev) // 2] * 1e3:.0f} us")
