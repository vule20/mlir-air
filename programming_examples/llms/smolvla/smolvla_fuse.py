# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the SigLIP layer is split into ELFs. One place, read by both the encoder
(what to compile and dispatch) and the runtime (which cache dir and which kernel
names to expect), so the two cannot disagree.

SMOLVLA_FUSE_FA:
  "0"      (default) three ELFs: vit_ln_qkv, flash_attn, vit_o_ffn. Both fused ELFs
           run with their loops fully unrolled (see OFFN_TILING / LNQKV_TILING).
  "1"      FlashAttention is the third launch of vit_ln_qkv, so a layer is two ELF
           dispatches. It saves a dispatch but the FA launch forces the tiling of
           the WHOLE merged ELF to all ones, so vit_ln_qkv loses its 12,6 tiling and
           the net is ~1 ms/image slower than "0".
  "layer"  experimental: the whole layer as one 9-launch ELF. Correct, but slower
           than "1" on NPU2 (FA's 3-D launch forces the o_ffn GEMMs off their
           [2,2] runtime-loop tiling).
"""

import os

# The LayerNorm implementation: the C++ row kernel (layer_norm_rows.cc, ~1.6x
# faster, same accuracy) unless SMOLVLA_LN_EXT=0 selects the air.api DSL loop.
LN_EXT = os.environ.get("SMOLVLA_LN_EXT", "1") == "1"
# runtime_loop_tiling_sizes of the vit_o_ffn ELF: how the launch-iteration loops of
# its GEMMs are split into unrolled tiles in the control stream. The deployed value
# was "2,2"; larger tiles unroll more and shrink the control stream the firmware has
# to parse (830 KB at 2,2, 715 KB from 6,6 up), which is most of the ELF penalty.
# 12,6 (= the launch grids' own extents) is 10% faster on the ELF and gives a
# bit-identical model output. The tiling is part of the cache key.
OFFN_TILING = [int(t) for t in os.environ.get("SMOLVLA_OFFN_TILING", "12,6").split(",")]
# The same for vit_ln_qkv, used only when FlashAttention is not merged into it: the
# FA launch needs all-ones tiling (see smolvla_vision_encoder), which drags the
# whole merged ELF down with it. 12,6 is 500 us faster per dispatch than 2,2.
LNQKV_TILING = [int(t) for t in os.environ.get("SMOLVLA_LNQKV_TILING", "12,6").split(",")]

# Rows one kernel call normalizes; the best point on NPU2 (layer_norm_rows.cc).
LN_ROWS = 4

FUSE_MODE = os.environ.get("SMOLVLA_FUSE_FA", "0")
assert FUSE_MODE in ("0", "1", "layer"), (
    f"SMOLVLA_FUSE_FA={FUSE_MODE!r}: expected '0', '1' or 'layer'"
)
