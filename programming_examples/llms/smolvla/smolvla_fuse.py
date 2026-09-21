# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the SigLIP layer is split into ELFs. One place, read by both the encoder
(what to compile and dispatch) and the runtime (which cache dir and which kernel
names to expect), so the two cannot disagree.

SMOLVLA_FUSE_FA:
  "1"      (default) FlashAttention is the third launch of vit_ln_qkv, so a layer is
           two ELF dispatches: vit_ln_qkv+FA, then vit_o_ffn.
  "0"      the original three ELFs: vit_ln_qkv, flash_attn, vit_o_ffn.
  "layer"  experimental: the whole layer as one 9-launch ELF. Correct, but slower
           than "1" on NPU2 (FA's 3-D launch forces the o_ffn GEMMs off their
           [2,2] runtime-loop tiling).
"""

import os

# The LayerNorm implementation: the C++ row kernel (layer_norm_rows.cc, ~1.6x
# faster, same accuracy) unless SMOLVLA_LN_EXT=0 selects the air.api DSL loop.
LN_EXT = os.environ.get("SMOLVLA_LN_EXT", "1") == "1"
# Rows one kernel call normalizes; the best point on NPU2 (layer_norm_rows.cc).
LN_ROWS = 4

FUSE_MODE = os.environ.get("SMOLVLA_FUSE_FA", "1")
assert FUSE_MODE in ("0", "1", "layer"), (
    f"SMOLVLA_FUSE_FA={FUSE_MODE!r}: expected '0', '1' or 'layer'"
)
