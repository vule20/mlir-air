# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Backbone-specific o_ffn variant: Gate+Up fused into ONE GEMM (measured
-10.4% device time vs 2 separate launches, see
Vu_exp/smolvla_backbone_perf/BACKBONE_PROFILE.md "QKV/GateUp fusion A/B").

7 launches instead of shared/builders/o_ffn_multi.py's 8 (O, Residual-Add,
FFN-RMSNorm, GateUp-fused-GEMM, SwiGLU-from-wide, Down, FFN-Add). O, Residual,
RMSNorm, Down, FFN-Add are UNCHANGED, reused as-is from o_ffn_multi.py -- this
file only replaces Gate-GEMM + Up-GEMM + SwiGLU (3 launches -> 2).

New isolated file: does NOT touch shared/infra/stitching.py or
shared/builders/o_ffn_multi.py, so llama32_1b/smollm2_1_7b/qwen builds are
untouched. The custom SwiGLU-from-wide builder reuses o_ffn_multi.py's own
build_padded_add pattern (row-iterate, read a column-slice of one wide buffer
via `A[r, lo:hi]`) -- already proven correct there for a padded residual add,
same trick applied to the SiLU*mul activation instead.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROG = _HERE.parent.parent.parent
for p in (str(_PROG), str(_HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_dtypes import bfloat16
from air import api as air
from air.api import ops
from air.api.types import i32
from shared.builders.rms_gemms_rope_multi import _api_dtype
from shared.builders.o_ffn_multi import _build_add_2d_to_2d, _build_add_2d_to_1d
from shared.infra.stitching import _wrap_ir_in_launch, stitch_elf, KernelSlice, FuncArg, alloc_gemm_scratch


def _build_swiglu_from_wide(rows, hidden_dim, np_dtype, herd_x=8, herd_y=1, target="npu2"):
    """SiLU(gate) * up, reading gate/up as column-slices of ONE wide
    (rows, 2*hidden_dim) buffer instead of two separate tensors -- same
    row-iterate + column-slice pattern as o_ffn_multi.py's build_padded_add,
    applied to the silu_and_mul extern instead of an add.
    """
    total_tiles = herd_x * herd_y
    assert rows % total_tiles == 0, (rows, total_tiles)
    rows_per_tile = rows // total_tiles

    dtype = _api_dtype(np_dtype)
    GATE_UP = air.tensor([rows, 2 * hidden_dim], dtype)
    OUT = air.tensor([rows, hidden_dim], dtype)

    activation = air.extern("silu_and_mul_bf16", link_with="silu_and_mul.o", scalars=[i32])

    with air.launch(name="swiglu_from_wide") as launch:

        @launch.body
        def _():
            with air.segment(name="swiglu_wide_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_x), range(herd_y)], name="swiglu_wide_herd", shape=(herd_x, herd_y)
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            l1_gate = air.alloc([hidden_dim], dtype, scope=h.private())
                            l1_up = air.alloc([hidden_dim], dtype, scope=h.private())
                            l1_out = air.alloc([hidden_dim], dtype, scope=h.private())

                            for iv in air.sequential(0, rows_per_tile):
                                r = (tx * herd_y + ty) * rows_per_tile + iv
                                ops.load(l1_gate, GATE_UP[r, 0:hidden_dim])
                                ops.load(l1_up, GATE_UP[r, hidden_dim : 2 * hidden_dim])
                                activation(l1_gate, l1_up, l1_out, hidden_dim)
                                ops.store(l1_out, OUT[r, 0:hidden_dim])

    return launch.build(target=target)


def build_o_ffn_module_fused_gu(seq_len, emb_dim, hidden_dim, herd_m=4, herd_n=4, print_kernels=False, gu_tile_n=80):
    """O-proj + Residual + FFN with Gate+Up fused into one GEMM.

    7 launches, args (base, before scratch tail):
      %arg0  attn_out    (seq_len, emb_dim)              O-GEMM input
      %arg1  wo          (emb_dim, emb_dim)
      %arg2  proj        (seq_len, emb_dim)               O-GEMM output
      %arg3  x_residual  (seq_len, emb_dim)
      %arg4  res1        (seq_len, emb_dim)               residual output (shared w/ FFN-Add)
      %arg5  ffn_norm_w  (emb_dim,)
      %arg6  normed2     (seq_len, emb_dim)                FFN RMSNorm output
      %arg7  w_gateup    (emb_dim, 2*hidden_dim)           [Wgate|Wup] concatenated
      %arg8  gate_up     (seq_len, 2*hidden_dim)           fused GEMM output
      %arg9  swiglu      (seq_len, hidden_dim)             SwiGLU-from-wide output
      %arg10 w_down      (hidden_dim, emb_dim)
      %arg11 down        (seq_len, emb_dim)                Down-GEMM output
      %arg12 output      (seq_len*emb_dim,)                FFN Add output
    """
    from shared.builders.gemm_builder import _build_gemm_module, gemm_registry_config, disambiguate_by_tile_n
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    o_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    d_spec = gemm_registry_config(seq_len, hidden_dim, emb_dim, "bf16", "high")
    o_spec, d_spec = disambiguate_by_tile_n([o_spec, d_spec])

    def _tiles(spec):
        return (dict(spec["build_kwargs"]), spec["tile_m"], spec["tile_k_l2"], spec["tile_k_l1"], spec["tile_n"])

    _o_kw, _o_m, _o_k2, _o_k1, _o_n = _tiles(o_spec)
    _d_kw, _d_m, _d_k2, _d_k1, _d_n = _tiles(d_spec)

    n_total = seq_len * emb_dim

    print("  [1/7] O GEMM (drain)...")
    o_ir = str(_build_gemm_module(seq_len, emb_dim, emb_dim, _o_m, _o_k2, _o_k1, _o_n, herd_m, herd_n, **_o_kw))

    print("  [2/7] Residual Add (2D -> 2D)...")
    res_add_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))

    print("  [3/7] FFN RMSNorm...")
    rms_ir = _wrap_ir_in_launch(str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8)))

    # Fused GateUp GEMM: same tiles validated in the standalone A/B
    # (backbone_qkv_fusion_ab.py FUSED_GU: -10.4% vs 2 separate launches).
    gu_n = 2 * hidden_dim
    gu_tile_k1 = 32
    print("  [4/7] GateUp GEMM, fused (drain)...")
    from shared.infra.external_kernels import compile_gemm_mm

    compile_gemm_mm(tile_m=32, tile_n=gu_tile_n, tile_k_l1=gu_tile_k1, sym_suffix="_gu", out_name="mm_gu.o")
    gu_ir = str(
        _build_gemm_module(
            seq_len, emb_dim, gu_n, 32, emb_dim, gu_tile_k1, gu_tile_n, herd_m, herd_n,
            external_bf16_out=True, sym_suffix="_gu", link_with_name="mm_gu.o",
        )
    )

    print("  [5/7] SwiGLU (from fused GateUp buffer)...")
    swiglu_ir = _wrap_ir_in_launch(str(_build_swiglu_from_wide(seq_len, hidden_dim, bfloat16, herd_x=8)))

    print("  [6/7] Down GEMM (drain)...")
    down_ir = str(_build_gemm_module(seq_len, hidden_dim, emb_dim, _d_m, _d_k2, _d_k1, _d_n, herd_m, herd_n, **_d_kw))

    print("  [7/7] FFN Add (2D -> 1D)...")
    ffn_add_ir = str(_build_add_2d_to_1d(seq_len, emb_dim, bfloat16))

    if print_kernels:
        for name, ir in [
            ("O GEMM", o_ir), ("Res Add", res_add_ir), ("FFN RMSNorm", rms_ir),
            ("GateUp GEMM", gu_ir), ("SwiGLU-wide", swiglu_ir), ("Down GEMM", down_ir), ("FFN Add", ffn_add_ir),
        ]:
            print(f"\n{'='*60}\n  Sub-kernel: {name} ({len(ir.splitlines())} lines)\n{'='*60}")
            print(ir)

    def _gemm_extern_syms(spec):
        sfx = spec["sym_suffix"]
        return {"@matmul_bf16", "@op_has_no_registered_library_name" + sfx, "@zero_f32_mn" + sfx, "@f32_to_bf16_mn" + sfx}

    def _gemm_arg_map(in_idx, w_idx, out_idx, sc):
        if sc is not None:
            return {0: in_idx, 1: w_idx, 2: sc, 3: out_idx}
        return {0: in_idx, 1: w_idx, 2: out_idx}

    gu_extern_syms = {"@matmul_bf16", "@op_has_no_registered_library_name_gu", "@zero_f32_mn_gu", "@f32_to_bf16_mn_gu"}

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{emb_dim}x{gu_n}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{gu_n}xbf16>"),
        FuncArg("%arg9", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{hidden_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{n_total}xbf16>"),
    ]
    scratch_args, scratch_for = alloc_gemm_scratch(
        [(o_spec, seq_len, emb_dim), (d_spec, seq_len, emb_dim)], base_arg_count=13,
    )

    slices = [
        KernelSlice(o_ir, "og", _gemm_arg_map(0, 1, 2, scratch_for[0]), extern_syms=_gemm_extern_syms(o_spec)),
        KernelSlice(res_add_ir, "ra", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(rms_ir, "rm", {0: 4, 1: 5, 2: 6}, private_from=False),
        KernelSlice(gu_ir, "gu", {0: 6, 1: 7, 2: 8}, extern_syms=gu_extern_syms),
        KernelSlice(swiglu_ir, "sw", {0: 8, 1: 9}, extern_syms={"@silu_and_mul_bf16"}),
        KernelSlice(down_ir, "dg", _gemm_arg_map(9, 10, 11, scratch_for[1]), extern_syms=_gemm_extern_syms(d_spec)),
        KernelSlice(ffn_add_ir, "fa", {0: 11, 1: 4, 2: 12}, private_from=False),
    ]

    module = stitch_elf("o_ffn_fused_gu", base_args, slices, scratch_args=scratch_args)
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
    return module
