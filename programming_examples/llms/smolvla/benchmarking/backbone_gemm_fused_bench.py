# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fused-ELF prototype for the SmolLM2-360M backbone's per-layer GEMMs, at the
real shape (M=241 padded to 256, hidden=960, GQA 15Q/5KV head_dim=64, MLP
mid=2560). Same method as expert_gemm_fused_bench2.py -- one multi-launch ELF
per layer via shared/infra/stitching.py (the machinery vit_o_ffn uses),
independent random per-GEMM data (no residual/RMSNorm/RoPE/attention -- this
isolates dispatch-fusion cost for the GEMMs only, exactly the expert bench's
scope). See Vu_exp/smolvla_backbone_perf/BACKBONE_PROFILE.md for the CPU
baseline this compares against and why M=241 (vs the expert's M=50) is the
open question.

Usage: python backbone_gemm_fused_bench.py [--iters 100] [--warmup 20] [--verify]
"""
from __future__ import annotations
import argparse, os, sys, time
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

SEQ = 241  # backbone prefix length
M = 256  # 241 padded to a 32-multiple
HIDDEN, INTER = 960, 2560
HEADS, KV_HEADS, HEAD_DIM = 15, 5, 64
Q_DIM, KV_DIM = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM  # 960, 320

TILE_M0, TILE_N, HERD_N = 32, 96, 4


def pad_up(n, block):
    return ((n + block - 1) // block) * block


def herd_m_for(m):
    hm = m // TILE_M0
    assert m % TILE_M0 == 0 and hm >= 1
    return hm


# name: (M, K, N, tile_k_l1, tile_k_l2). qkv and gate+up fused each into one
# GEMM (same as expert bench's "qkv"/"gu"), matching the real weight layout
# (separate q_proj/k_proj/v_proj and gate_proj/up_proj columns concatenated).
BACKBONE_GEMMS = [
    ("qkv", M, HIDDEN, Q_DIM + 2 * KV_DIM, 96, HIDDEN),  # 960 -> 1600
    ("o", M, Q_DIM, HIDDEN, 96, Q_DIM),  # 960 -> 960
    ("gu", M, HIDDEN, 2 * INTER, 96, HIDDEN),  # 960 -> 5120
    ("dp", M, INTER, HIDDEN, 64, 256),  # 2560 -> 960
]


def build_fused(tag, gemms, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    base_args, slices, shapes = [], [], []
    for i, (name, m, k, n, tk1, tk2) in enumerate(gemms):
        n_pad = pad_up(n, TILE_N * HERD_N)
        hm = herd_m_for(m)
        sfx = f"_{tag}_{name}"
        out_name = f"mm{sfx}.o"
        compile_gemm_mm(tile_m=TILE_M0, tile_n=TILE_N, tile_k_l1=tk1, sym_suffix=sfx, out_name=out_name)
        ir = str(
            _build_gemm_module(
                m, k, n_pad, TILE_M0, tk2, tk1, TILE_N, hm, HERD_N,
                external_bf16_out=True, sym_suffix=sfx, link_with_name=out_name,
            )
        )
        a0 = 3 * i
        base_args += [
            FuncArg(f"%arg{a0}", f"memref<{m}x{k}xbf16>"),
            FuncArg(f"%arg{a0 + 1}", f"memref<{k}x{n_pad}xbf16>"),
            FuncArg(f"%arg{a0 + 2}", f"memref<{m}x{n_pad}xbf16>"),
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
        shapes.append((name, m, k, n, n_pad))
    mod = stitch_elf(f"backbone_{tag}_fused", base_args, slices)

    from air.backend.xrt import XRTBackend
    backend = XRTBackend(
        verbose=False, omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[2, 2], stack_size=2048,
        output_format="elf", instance_name=f"backbone_{tag}_fused", target_device="npu2",
        n_perf_iters=0,
    )
    artifact = backend.compile(mod)
    return backend, artifact, shapes


def time_fused(tag, gemms, warmup, iters, verify):
    import filelock, pyxrt as xrt

    backend, artifact, shapes = build_fused(tag, gemms, _HERE / "build" / f"backbone_gemm_fused_{tag}")

    rng = np.random.default_rng(hash(tag) & 0xFFFF)
    args_np, sizes, ref_pairs = [], [], []
    for name, m, k, n, n_pad in shapes:
        A = rng.integers(-4, 4, size=(m, k)).astype(bfloat16)
        B = np.zeros((k, n_pad), dtype=bfloat16)
        B[:, :n] = rng.integers(-4, 4, size=(k, n)).astype(bfloat16)
        C = np.zeros((m, n_pad), dtype=bfloat16)
        args_np += [A, B, C]
        sizes += [a.size * a.itemsize for a in (A, B, C)]
        ref_pairs.append((A, B, n))

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

        completed = getattr(getattr(xrt, "ert_cmd_state", None), "ERT_CMD_STATE_COMPLETED", None)

        def launch():
            run = xrt.run(backend.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            run.start()
            st = run.wait2()
            if st is not None and completed is not None and st != completed:
                raise RuntimeError(f"NPU run failed: {st}")

        def read_all():
            for i in range(2, len(bos), 3):
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            return [
                np.frombuffer(bo_maps[i].tobytes(), dtype=bfloat16).reshape(args_np[i].shape)
                for i in range(2, len(bos), 3)
            ]

        kernel_us = []
        for it in range(warmup + iters):
            write_inputs()
            t0 = time.perf_counter()
            launch()
            t1 = time.perf_counter()
            _ = read_all()
            if it >= warmup:
                kernel_us.append((t1 - t0) * 1e6)

        cosines = None
        if verify:
            write_inputs()
            launch()
            outs = read_all()
            cosines = []
            for (A, B, n), C in zip(ref_pairs, outs):
                out = C[:, :n].astype(np.float32)
                ref = A.astype(np.float32) @ B[:, :n].astype(np.float32)
                cosines.append(
                    float(
                        np.dot(out.ravel(), ref.ravel())
                        / (np.linalg.norm(out.ravel()) * np.linalg.norm(ref.ravel()) + 1e-9)
                    )
                )
        backend.unload()

    avg = sum(kernel_us) / len(kernel_us)
    return avg, min(kernel_us), max(kernel_us), cosines, [s[0] for s in shapes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    avg, mn, mx, cosines, names = time_fused("layer", BACKBONE_GEMMS, args.warmup, args.iters, args.verify)
    print(f"1 backbone layer, 1 fused ELF ({'+'.join(names)}): avg={avg:.1f}us min={mn:.1f}us max={mx:.1f}us")
    if cosines:
        for n, c in zip(names, cosines):
            print(f"    {n:4s} cosine={c:.6f}")

    print(f"\n16-layer backbone, GEMMs only: {16 * avg / 1000:.2f} ms")
    print("CPU baseline (Vu_exp/smolvla_backbone_perf/BACKBONE_PROFILE.md): GEMMs only 26.564 ms, full fill 43.9 ms")


if __name__ == "__main__":
    main()
