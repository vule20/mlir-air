# SPDX-License-Identifier: MIT
"""One whole SmolVLA backbone layer as a single multi-launch ELF.

Stitches the three per-layer ELFs -- rms_gemms_rope_fused_qkv (4 launches),
masked FlashAttention (1 launch) and o_ffn_fused_gu (6 launches) -- into one
func, so a layer is one XRT dispatch instead of three. Each sub-module's
launches keep their own shim-DMA tiling through the per-launch
`air.shim_dma_tile_sizes` attribute: the global runtime_loop_tiling_sizes
cannot serve all three (the FA head axis only tolerates a factor of 1).

Combined args:
  %arg0  x           (seq, emb)          layer input, also the O residual
  %arg1  attn_norm   (emb,)
  %arg2  normed      (seq, emb)
  %arg3  w_qkv       (emb, emb+2kv)
  %arg4  qkv         (seq, emb+2kv)      FA reads V from its last kv columns
  %arg5  rope_lut_q
  %arg6  q_roped     (seq, emb)
  %arg7  rope_lut_k
  %arg8  k_roped     (seq, kv)
  %arg9  mask        (seq, seq)          additive bf16
  %arg10 attn_out    (seq, emb)
  %arg11 wo          (emb, emb)
  %arg12 proj        (seq, emb)
  %arg13 res1        (seq, emb)
  %arg14 ffn_norm    (emb,)
  %arg15 normed2     (seq, emb)
  %arg16 w_gateup    (emb, 2*hidden)     SwiGLU-interleaved
  %arg17 swiglu      (seq, hidden)
  %arg18 w_down      (hidden, emb)
  %arg19 down        (seq, emb)
  %arg20 output      (seq*emb,)
"""
import re

from shared.infra.stitching import FuncArg, KernelSlice, stitch_elf

# Sub-module operand -> combined arg. o_ffn's operand 8 (the wide gate|up
# buffer) is dead under the SwiGLU epilogue and is dropped.
_RGR_MAP = {i: i for i in range(9)}
_FA_MAP = {0: 6, 1: 8, 2: 4, 3: 9, 4: 10}
_OFFN_MAP = {0: 10, 1: 11, 2: 12, 3: 0, 4: 13, 5: 14, 6: 15, 7: 16, 9: 17, 10: 18, 11: 19, 12: 20}

LAYER_STATIC = {1, 3, 5, 7, 9, 11, 14, 16, 18}
LAYER_INTERMEDIATE = {2, 4, 6, 8, 10, 12, 13, 15, 17, 19, 20}
LAYER_OUT = 20


def _signature_types(ir):
    sig = re.search(r"func\.func @\w+\(([^)]*)\)", ir).group(1)
    return [a.split(":", 1)[1].strip() for a in sig.split(",") if a.strip()]


def _privates(ir):
    return set(re.findall(r"func\.func private (@\w+)", ir))


def build_layer_module(rgr_ir, fa_ir, offn_ir, tilings):
    """rgr_ir / fa_ir / offn_ir: sub-module texts (FA built with attn_mask=True
    and v_cols = the qkv width). tilings: {"rgr", "fa", "offn"} -> the shim-DMA
    tile sizes each sub-module's launches were tuned with."""
    types = [None] * 21
    for ir, amap in ((rgr_ir, _RGR_MAP), (fa_ir, _FA_MAP), (offn_ir, _OFFN_MAP)):
        for op_idx, t in enumerate(_signature_types(ir)):
            if op_idx in amap:
                c = amap[op_idx]
                assert types[c] in (None, t), f"arg{c}: {types[c]} vs {t}"
                types[c] = t
    assert None not in types, types
    base_args = [FuncArg(f"%arg{i}", t) for i, t in enumerate(types)]

    parts = (("rg", rgr_ir, _RGR_MAP, "rgr"), ("at", fa_ir, _FA_MAP, "fa"), ("of", offn_ir, _OFFN_MAP, "offn"))
    slices = [KernelSlice(ir, p, amap, extern_syms=_privates(ir)) for p, ir, amap, _ in parts]
    module = stitch_elf("layer", base_args, slices, debug_dump_path="/tmp/layer_fused_parse_error.mlir")

    from air.ir import DenseI64ArrayAttr

    per_launch = [tilings[key] for _, ir, _, key in parts for _ in range(ir.count("air.launch "))]
    func = next(
        op for op in module.body.operations
        if op.operation.name == "func.func" and op.attributes["sym_name"].value == "layer"
    )
    launches = [op for op in func.regions[0].blocks[0].operations if op.operation.name == "air.launch"]
    assert len(launches) == len(per_launch), (len(launches), len(per_launch))
    with module.context:
        for op, ts in zip(launches, per_launch):
            op.attributes["air.shim_dma_tile_sizes"] = DenseI64ArrayAttr.get(ts)
    print(f"  Layer module: {len(launches)} launches, {len(str(module).splitlines())} lines, parsed OK")
    return module
