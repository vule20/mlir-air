"""Run the SmolVLA backbone's transformer layers on NPU2 via llama32_1b's
already-built, already-fused prefill machinery (rms_gemms_rope + flash_attn/
CPU-attn + o_ffn per layer), instead of the hand-stitched raw-GEMM prototype in
backbone_gemm_fused_bench.py.

Backbone's `text_model` is confirmed to BE `transformers.models.llama.modeling_llama
.LlamaModel` (bit-for-bit, not "llama-like") -- see Vu_exp/smolvla_backbone_perf/
BACKBONE_PROFILE.md. But lerobot's SmolVLMWithExpertModel.forward_attn_layer does
NOT call LlamaModel.forward(): it reaches into layer.self_attn.{q,k,v,o}_proj and
layer.mlp directly and re-implements RoPE/masking/residuals itself, with its own
conventions that differ from the HF config:
  - RoPE base = 10000 (apply_rope's hardcoded default), NOT text_config.rope_theta
    (100000) -- confirmed by reading every apply_rope call site, none override it.
  - position_ids are NOT arange(seq_len): they REPEAT (multi-camera prefix tokens
    share position ids), so the RoPE LUT must be gathered by the real position_id
    values, not indexed by sequence position.
  - attention_mask is a real (seq, seq) bool prefix mask (confirmed non-causal,
    non-symmetric by probing it), not causal -- llama32_1b's attention_reference
    hardcodes a causal mask, so it is monkeypatched here with a masked variant.
  - RMSNorm eps=1e-5 matches llama32_1b_cpu_helpers' default; no change needed.

Usage: python backbone_npu.py [--layers 16] [--cpu-attn] [--profile]
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_SMOLVLA = Path(__file__).resolve().parent.parent
_LLMS = _SMOLVLA.parent
_LLAMA = _LLMS / "llama32_1b"
for p in (str(_SMOLVLA), str(_LLMS), str(_LLAMA)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("SMOLVLA_CPU_BIND", "1")
os.environ.setdefault("SMOLVLA_CPU_THREADS", "8")
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

import llama32_1b_prefill as prefill  # noqa: E402
from llama32_1b_weights import LlamaConfig, LlamaWeights, LayerWeights  # noqa: E402

BACKBONE_CONFIG = LlamaConfig(
    n_layers=16,
    emb_dim=960,
    n_heads=15,
    head_dim=64,
    n_kv_heads=5,
    hidden_dim=2560,
    vocab_size=1,  # unused: backbone forward never touches embed/lm_head here
    rope_base=10000.0,  # apply_rope's hardcoded default, NOT text_config.rope_theta=100000
)

SEQ_REAL = 241
SEQ_PAD = 256  # 241 padded to a multiple of 64 (llama32_1b_prefill's fused-cast GEMM requirement)


def extract_backbone_weights(policy) -> LlamaWeights:
    """Pull the REAL loaded weights straight out of the live nn.Module (not a
    fresh safetensors download) -- guarantees an exact match with the CPU
    reference for correctness comparison."""
    import torch

    tm = policy.model.vlm_with_expert.get_vlm_model().text_model
    layers = []
    for layer in tm.layers:
        t = lambda w: np.ascontiguousarray(w.detach().to(torch.float32).numpy().T).astype(bfloat16)
        norm = lambda w: w.detach().to(torch.float32).numpy().astype(bfloat16)
        layers.append(
            LayerWeights(
                attn_norm=norm(layer.input_layernorm.weight),
                wq=t(layer.self_attn.q_proj.weight),
                wk=t(layer.self_attn.k_proj.weight),
                wv=t(layer.self_attn.v_proj.weight),
                wo=t(layer.self_attn.o_proj.weight),
                ffn_norm=norm(layer.post_attention_layernorm.weight),
                w_gate=t(layer.mlp.gate_proj.weight),
                w_up=t(layer.mlp.up_proj.weight),
                w_down=t(layer.mlp.down_proj.weight),
            )
        )
    return LlamaWeights(embed_table=None, layers=layers, final_norm=None, lm_head=None)


def build_rope_lut_gathered(position_ids: np.ndarray, config: LlamaConfig) -> np.ndarray:
    """LUT rows gathered by the REAL position_id values (which repeat), not by
    sequence index -- matches apply_rope's `radians = positions / timescale`
    exactly (verified algebraically identical to standard rotate_half RoPE)."""
    head_dim = config.head_dim
    half = head_dim // 2
    max_pos = int(position_ids.max()) + 1
    dim_indices = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (config.rope_base ** (dim_indices / head_dim))
    positions = np.arange(max_pos, dtype=np.float64)
    angles = np.outer(positions, inv_freq)
    base_lut = np.empty((max_pos, head_dim), dtype=np.float64)
    base_lut[:, :half] = np.cos(angles)
    base_lut[:, half:] = np.sin(angles)
    return base_lut[position_ids].astype(bfloat16)


def masked_attention_reference(q, k, v, n_heads, n_kv_heads, mask_bool):
    """Same structure as llama32_1b_cpu_helpers.attention_reference, but with
    an explicit (seq, seq) bool mask (True=attend) instead of a hardcoded
    causal mask -- matches eager_attention_forward's `torch.where(mask, w, big_neg)`."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    seq_len = q.shape[0]
    head_dim = q.shape[1] // n_heads
    group_size = n_heads // n_kv_heads

    q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(head_dim)
    big_neg = np.finfo(np.float32).min
    add_mask = np.where(mask_bool, 0.0, big_neg).astype(np.float32)

    out_heads = np.empty((n_heads, seq_len, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_idx = h // group_size
        scores = q[h] @ k[kv_idx].T * scale + add_mask
        m = scores.max(axis=-1, keepdims=True)
        p = np.exp(scores - m)
        probs = p / p.sum(axis=-1, keepdims=True)
        out_heads[h] = probs @ v[kv_idx]
    return out_heads.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)


_MASK_HOLDER = {"mask": None}


def _patched_attention_reference(q, k, v, n_heads, n_kv_heads):
    return masked_attention_reference(q, k, v, n_heads, n_kv_heads, _MASK_HOLDER["mask"])


def capture_real_backbone_io(policy, n_layers_to_capture):
    """Hook forward_attn_layer (gives real hidden-in/mask/position_ids per
    layer) and o_proj/mlp (gives the pieces needed to reconstruct the exact
    per-layer output, since lerobot inlines the residual adds in plain Python
    rather than calling a single hookable `layer.forward`)."""
    import torch
    from smolvla_inference import build_oracle_batch, fixed_noise, run_hybrid_forward

    vwe = policy.model.vlm_with_expert
    orig_fal = vwe.forward_attn_layer
    per_layer = {}

    def hooked_fal(model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, *rest, **kw):
        if layer_idx < n_layers_to_capture and layer_idx not in per_layer:
            per_layer[layer_idx] = {
                "hidden_in": inputs_embeds[0].detach().clone(),
                "position_ids": position_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
            }
        return orig_fal(model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, *rest, **kw)

    vwe.forward_attn_layer = hooked_fal

    # o_proj / mlp hooks to reconstruct the real per-layer output (see forward()
    # lines ~480-491: out_emb = o_proj(att_out) + hidden_in; after_first_residual
    # = out_emb; out_emb = mlp(post_attention_layernorm(out_emb)); out_emb += after_first_residual).
    tm = policy.model.vlm_with_expert.get_vlm_model().text_model
    o_outs, mlp_outs = {}, {}
    o_hooks, mlp_hooks = [], []
    call_idx = {"o": 0, "mlp": 0}

    def make_o_hook(idx):
        def hook(module, inp, out):
            if call_idx["o"] == idx:
                o_outs[idx] = out.detach().clone()
            call_idx["o"] += 1
        return hook

    def make_mlp_hook(idx):
        def hook(module, inp, out):
            if call_idx["mlp"] == idx:
                mlp_outs[idx] = out.detach().clone()
            call_idx["mlp"] += 1
        return hook

    for i in range(n_layers_to_capture):
        o_hooks.append(tm.layers[i].self_attn.o_proj.register_forward_hook(make_o_hook(i)))
        mlp_hooks.append(tm.layers[i].mlp.register_forward_hook(make_mlp_hook(i)))

    policy_eval = policy
    batch = build_oracle_batch(policy_eval, n_cameras=3)
    noise = fixed_noise(policy_eval)
    run_hybrid_forward(batch, policy=policy_eval, noise=noise, npu_vision=False)

    vwe.forward_attn_layer = orig_fal
    for h in o_hooks + mlp_hooks:
        h.remove()

    real_outputs = {}
    for i in range(n_layers_to_capture):
        hidden_in = per_layer[i]["hidden_in"]
        after_first_residual = o_outs[i] + hidden_in
        real_outputs[i] = (
            (after_first_residual + mlp_outs[i]).squeeze(0).to(torch.float32).numpy()
        )
        per_layer[i]["hidden_in"] = hidden_in.squeeze(0).to(torch.float32).numpy()
        per_layer[i]["position_ids"] = per_layer[i]["position_ids"].squeeze(0).numpy()
        per_layer[i]["attention_mask"] = per_layer[i]["attention_mask"].squeeze(0).numpy()

    return per_layer, real_outputs


def compile_backbone_kernels(
    cache, config, seq_len, herd_m_override=None, fused_gu=False, gu_tile_n=80,
    fused_qkv=False, qkv_tile_n=80, gu_bstationary=False, qkv_bstationary=False,
    od_bstationary=False, od_tile_n=80,
):
    """Replacement for llama32_1b_prefill.compile_all_kernels: that function
    hardcodes mm.o pre-compiles at tile_n=128 (llama32_1b's own registry
    tile_n), which would SILENTLY produce wrong results here -- backbone's
    registry rows resolve to tile_n=80 (Q/K/V/O/Down) and tile_n=128
    (Gate/Up), and compile_gemm_mm bakes tile_n as a compile-time DIM_N macro
    into the object; reusing a wrongly-baked mm_m32.o under the same symbol
    name is not a link error, it is a silent correctness bug. Verified via a
    probe (not guessed) exactly which 3 (tile_m, tile_n, tile_k_l1, sym_suffix)
    combinations the registry + disambiguate_by_tile_n actually produce for
    this config at seq_len=256:
        rms_gemms_rope Q/K/V : drain, tile_n=80,  sym_suffix "_m32"      -> mm_m32.o
        o_ffn O/Down         : drain, tile_n=80,  sym_suffix "_m32_n80"  -> mm_m32_n80.o
        o_ffn Gate/Up        : drain, tile_n=128, sym_suffix "_m32_n128"-> mm_m32_n128.o
    """
    from shared.infra.external_kernels import compile_gemm_mm
    from shared.builders.rms_gemms_rope_multi import build_rms_gemms_rope_module
    from shared.builders.o_ffn_multi import build_o_ffn_module

    compile_gemm_mm(tile_m=32, tile_n=80, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o")
    compile_gemm_mm(tile_m=32, tile_n=80, tile_k_l1=32, sym_suffix="_m32_n80", out_name="mm_m32_n80.o")
    compile_gemm_mm(tile_m=32, tile_n=128, tile_k_l1=32, sym_suffix="_m32_n128", out_name="mm_m32_n128.o")

    gemm_herd_m = herd_m_override or next(h for h in (8, 4, 2, 1) if seq_len % (64 * h) == 0)

    if fused_qkv:
        from rms_gemms_rope_fused_qkv import build_rms_gemms_rope_module_fused_qkv

        cache.compile_and_cache(
            "rms_gemms_rope",
            build_rms_gemms_rope_module_fused_qkv(
                seq_len, config.emb_dim, config.n_kv_heads * config.head_dim,
                config.n_heads, config.n_kv_heads, config.head_dim, herd_m=gemm_herd_m, qkv_tile_n=qkv_tile_n,
                b_stationary=qkv_bstationary,
            ),
            {"verbose": cache.verbose, "omit_while_true_loop": False, "output_format": "elf",
             "instance_name": "rms_gemms_rope_fused_qkv", "runtime_loop_tiling_sizes": [2, 2]},
        )
    else:
        cache.compile_and_cache(
            "rms_gemms_rope",
            build_rms_gemms_rope_module(
                seq_len, config.emb_dim, config.n_kv_heads * config.head_dim,
                config.n_heads, config.n_kv_heads, config.head_dim, herd_m=gemm_herd_m,
            ),
            {"verbose": cache.verbose, **prefill._rms_gemms_rope_run_backend()},
        )
    o_ffn_backend = {
        "verbose": cache.verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_ffn",
        "runtime_loop_tiling_sizes": [2, 2],
    }
    if fused_gu:
        from o_ffn_fused_gu import build_o_ffn_module_fused_gu

        cache.compile_and_cache(
            "o_ffn",
            build_o_ffn_module_fused_gu(
                seq_len, config.emb_dim, config.hidden_dim, herd_m=gemm_herd_m, gu_tile_n=gu_tile_n,
                gu_b_stationary=gu_bstationary, od_b_stationary=od_bstationary, od_tile_n=od_tile_n,
            ),
            {**o_ffn_backend, "instance_name": "o_ffn_fused_gu"},
        )
    else:
        cache.compile_and_cache(
            "o_ffn",
            build_o_ffn_module(seq_len, config.emb_dim, config.hidden_dim, herd_m=gemm_herd_m),
            o_ffn_backend,
        )
    cache._save_manifest()
    print(f"Backbone kernels compiled and cached to {cache.cache_dir}/")


def run_transformer_block_custom(
    x_bf16, layer_weights, rope_lut_bf16, config, cache, layer_idx=0, fused_qkv=False, fused_gu=False,
):
    """Same as llama32_1b_prefill.run_transformer_block, but either or both
    halves can use a fused-launch variant:
      fused_qkv: rms_gemms_rope_fused_qkv's 9-arg/4-launch RMS+QKV+RoPE
                 (Q+K+V in one GEMM; V has no RoPE, sliced host-side from the
                 fused GEMM's wide output) instead of the original 13-arg/6-launch.
      fused_gu:  o_ffn_fused_gu's 13-arg/7-launch O+FFN (Gate+Up in one GEMM)
                 instead of the original 15-arg/8-launch.
    Both False reproduces prefill.run_transformer_block's own arg layout
    exactly (kept here rather than delegating, so one function handles every
    combination without re-deriving the unfused arg lists twice).
    """
    seq_len = x_bf16.shape[0]
    emb_dim, n_heads, n_kv_heads, head_dim, hidden_dim = (
        config.emb_dim, config.n_heads, config.n_kv_heads, config.head_dim, config.hidden_dim
    )
    kv_dim = n_kv_heads * head_dim
    _arg_cache = getattr(run_transformer_block_custom, "_arg_cache", {})
    run_transformer_block_custom._arg_cache = _arg_cache

    # ---- RMS + QKV + RoPE ----
    if fused_qkv:
        qkv_n = emb_dim + 2 * kv_dim
        _rms_key = f"rms_gemms_rope_qkv_L{layer_idx}"
        if _rms_key not in _arg_cache:
            w_qkv = np.ascontiguousarray(
                np.concatenate([layer_weights.wq, layer_weights.wk, layer_weights.wv], axis=1)
            ).astype(bfloat16)
            _rms_args = [
                None,
                np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                w_qkv,
                np.zeros((seq_len, qkv_n), dtype=bfloat16),
                np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten(),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten(),
                np.zeros((seq_len, kv_dim), dtype=bfloat16),
            ]
            _arg_cache[_rms_key] = _rms_args
        cached_args = _arg_cache[_rms_key]
        cached_args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)

        results = cache.load_and_run(
            "rms_gemms_rope", {"verbose": False, "omit_while_true_loop": False, "output_format": "elf",
                               "instance_name": "rms_gemms_rope_fused_qkv", "runtime_loop_tiling_sizes": [2, 2]},
            *cached_args, output_indices=[4, 6, 8], static_input_indices={1, 3, 5, 7},
            intermediate_indices={2, 4, 6, 8}, bo_key=_rms_key, shared_nonstatic=True,
        )
        qkv_buf = results[4].reshape(seq_len, qkv_n)
        v = qkv_buf[:, emb_dim + kv_dim : emb_dim + 2 * kv_dim]
        q_roped = results[6].reshape(seq_len, emb_dim)
        k_roped = results[8].reshape(seq_len, kv_dim)
    else:
        _rms_key = f"rms_gemms_rope_L{layer_idx}"
        if _rms_key not in _arg_cache:
            _rms_args = [
                None,
                np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.asarray(layer_weights.wq, dtype=bfloat16).reshape(emb_dim, emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.asarray(layer_weights.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),
                np.zeros((seq_len, kv_dim), dtype=bfloat16),
                np.asarray(layer_weights.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),
                np.zeros((seq_len, kv_dim), dtype=bfloat16),
                np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten(),
                np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten(),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.zeros((seq_len, kv_dim), dtype=bfloat16),
            ]
            _scratch_arrays, _scratch_inter = prefill._rms_scratch_specs(seq_len, emb_dim, kv_dim)
            _rms_args.extend(_scratch_arrays)
            _arg_cache[_rms_key] = (_rms_args, _scratch_inter)
        cached_args, _scratch_inter = _arg_cache[_rms_key]
        cached_args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)

        _rms_inter = {2, 4, 6, 8, 11, 12} | _scratch_inter
        results = cache.load_and_run(
            "rms_gemms_rope", prefill._rms_gemms_rope_run_backend(), *cached_args,
            output_indices=[8, 11, 12], static_input_indices={1, 3, 5, 7, 9, 10},
            intermediate_indices=_rms_inter, bo_key=_rms_key, shared_nonstatic=True,
        )
        v = results[8].reshape(seq_len, kv_dim)
        q_roped = results[11].reshape(seq_len, n_heads * head_dim)
        k_roped = results[12].reshape(seq_len, n_kv_heads * head_dim)

    with cache.profiler.time_cpu("prefill_cpu_attention"):
        attn_out = prefill.attention_reference(
            q_roped.astype(np.float32), k_roped.astype(np.float32), v.astype(np.float32),
            n_heads, n_kv_heads,
        ).astype(bfloat16)

    # ---- O + Residual + FFN ----
    if fused_gu:
        gu_n = 2 * hidden_dim
        _offn_key = f"o_ffn_fused_gu_L{layer_idx}"
        if _offn_key not in _arg_cache:
            w_gateup = np.ascontiguousarray(
                np.concatenate([layer_weights.w_gate, layer_weights.w_up], axis=1)
            ).astype(bfloat16)
            offn_args = [
                None,
                np.asarray(layer_weights.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                None,
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                w_gateup,
                np.zeros((seq_len, gu_n), dtype=bfloat16),
                np.zeros((seq_len, hidden_dim), dtype=bfloat16),
                np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.zeros(seq_len * emb_dim, dtype=bfloat16),
            ]
            _arg_cache[_offn_key] = offn_args
        cached_args = _arg_cache[_offn_key]
        cached_args[0] = np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim)
        cached_args[3] = x_bf16.reshape(seq_len, emb_dim).astype(bfloat16, copy=False)

        _out_idx = 12
        _inter = {2, 4, 6, 8, 9, 11, 12}
        results = cache.load_and_run(
            "o_ffn", {"verbose": False, "omit_while_true_loop": False, "output_format": "elf",
                      "instance_name": "o_ffn_fused_gu", "runtime_loop_tiling_sizes": [2, 2]},
            *cached_args, output_indices=[_out_idx], static_input_indices={1, 5, 7, 10},
            intermediate_indices=_inter, bo_key=_offn_key, shared_nonstatic=True,
        )
        return results[_out_idx].reshape(seq_len, emb_dim)
    else:
        _offn_key = f"o_ffn_L{layer_idx}"
        if _offn_key not in _arg_cache:
            offn_args = [
                None,
                np.asarray(layer_weights.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                None,
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),
                np.zeros((seq_len, hidden_dim), dtype=bfloat16),
                np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),
                np.zeros((seq_len, hidden_dim), dtype=bfloat16),
                np.zeros((seq_len, hidden_dim), dtype=bfloat16),
                np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.zeros(seq_len * emb_dim, dtype=bfloat16),
            ]
            offn_args.extend(
                np.zeros(shape, dtype=np.float32) for shape in prefill._o_ffn_scratch_plan(seq_len, emb_dim, hidden_dim)[0]
            )
            _arg_cache[_offn_key] = offn_args
        cached_args = _arg_cache[_offn_key]
        cached_args[0] = np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim)
        cached_args[3] = x_bf16.reshape(seq_len, emb_dim).astype(bfloat16, copy=False)

        _out_idx = 14
        _inter = {2, 4, 6, 8, 10, 11, 13, 14} | prefill._o_ffn_scratch_plan(seq_len, emb_dim, hidden_dim)[1]
        results = cache.load_and_run(
            "o_ffn", prefill._o_ffn_run_backend(), *cached_args, output_indices=[_out_idx],
            static_input_indices={1, 5, 7, 9, 12}, intermediate_indices=_inter,
            bo_key=_offn_key, shared_nonstatic=True,
        )
        return results[_out_idx].reshape(seq_len, emb_dim)


def pad_seq(x, seq_pad, fill=0.0):
    out = np.full((seq_pad,) + x.shape[1:], fill, dtype=x.dtype)
    out[: x.shape[0]] = x
    return out


def pad_mask(mask_bool, seq_pad):
    """(seq_real, seq_real) -> (seq_pad, seq_pad): padding rows/cols blocked
    from real tokens (False), each padding row can see itself (diagonal True)
    so its own softmax stays finite -- its output is discarded regardless."""
    seq_real = mask_bool.shape[0]
    out = np.zeros((seq_pad, seq_pad), dtype=bool)
    out[:seq_real, :seq_real] = mask_bool
    for i in range(seq_real, seq_pad):
        out[i, i] = True
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--cpu-attn", action="store_true", default=True)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--herd-m", type=int, default=None)
    ap.add_argument("--fused-gu", action="store_true", help="Gate+Up fused into 1 GEMM (7 launches vs 8)")
    ap.add_argument("--gu-tile-n", type=int, default=80)
    ap.add_argument("--fused-qkv", action="store_true", help="Q+K+V fused into 1 GEMM (4 launches vs 6)")
    ap.add_argument("--qkv-tile-n", type=int, default=80)
    ap.add_argument("--gu-bstationary", action="store_true")
    ap.add_argument("--qkv-bstationary", action="store_true")
    ap.add_argument("--od-bstationary", action="store_true", help="O/Down GEMMs bypass registry, full-K + B-stationary")
    ap.add_argument("--od-tile-n", type=int, default=48)
    args = ap.parse_args()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval()
    print("Extracting real backbone weights...")
    weights = extract_backbone_weights(policy)

    print(f"Capturing real per-layer I/O for {args.layers} layer(s)...")
    per_layer, real_outputs = capture_real_backbone_io(policy, args.layers)

    layer0 = per_layer[0]
    print(f"hidden_in: {layer0['hidden_in'].shape}, position_ids: {layer0['position_ids'].shape}, "
          f"mask: {layer0['attention_mask'].shape}")

    hm_tag = f"_hm{args.herd_m}" if args.herd_m else ""
    gu_tag = f"_fgu{args.gu_tile_n}{'bst' if args.gu_bstationary else ''}{'od' if args.od_bstationary else ''}" if args.fused_gu else ""
    qkv_tag = f"_fqkv{args.qkv_tile_n}{'bst' if args.qkv_bstationary else ''}" if args.fused_qkv else ""
    cache_dir = str(Path(__file__).resolve().parent / "build" / f"backbone_npu_cache{hm_tag}{gu_tag}{qkv_tag}")
    from shared.infra.cache import KernelCache, Profiler

    cache = KernelCache(cache_dir, verbose=False, profiler=Profiler(enabled=True))
    print(f"Compiling kernels (seq_len=256, herd_m={args.herd_m or 'auto'}, "
          f"fused_gu={args.fused_gu}, fused_qkv={args.fused_qkv})...")
    compile_backbone_kernels(
        cache, BACKBONE_CONFIG, SEQ_PAD, herd_m_override=args.herd_m,
        fused_gu=args.fused_gu, gu_tile_n=args.gu_tile_n,
        fused_qkv=args.fused_qkv, qkv_tile_n=args.qkv_tile_n,
        gu_bstationary=args.gu_bstationary, qkv_bstationary=args.qkv_bstationary,
        od_bstationary=args.od_bstationary, od_tile_n=args.od_tile_n,
    )
    prefill.attention_reference = _patched_attention_reference

    mask_padded = pad_mask(layer0["attention_mask"].astype(bool), SEQ_PAD)
    _MASK_HOLDER["mask"] = mask_padded

    x = pad_seq(layer0["hidden_in"].astype(bfloat16), SEQ_PAD)
    rope_lut_real = build_rope_lut_gathered(layer0["position_ids"], BACKBONE_CONFIG)
    # Pad positions: repeat the last real position id's LUT row for padding rows
    # (their output is discarded, but rms_gemms_rope indexes rope_lut[:seq_len]
    # contiguously so it needs SEQ_PAD rows).
    rope_lut_padded = pad_seq(rope_lut_real, SEQ_PAD, fill=0.0)
    rope_lut_padded[SEQ_REAL:] = rope_lut_real[-1]

    def run_one_layer(xx, i, verbose=False):
        if args.fused_gu or args.fused_qkv:
            return run_transformer_block_custom(
                xx, weights.layers[i], rope_lut_padded, BACKBONE_CONFIG, cache, layer_idx=i,
                fused_qkv=args.fused_qkv, fused_gu=args.fused_gu,
            )
        out, _inter = prefill.run_transformer_block(
            xx, weights.layers[i], rope_lut_padded, BACKBONE_CONFIG, cache,
            layer_idx=i, cpu_attn=True, verbose=verbose,
        )
        return out

    npu_outputs = {}
    for i in range(args.layers):
        out = run_one_layer(x, i, verbose=(i == 0))
        npu_outputs[i] = out
        x = out

    # Correctness: compare layer 0's NPU-path output (still bit-identical-ish
    # modulo bf16) against the REAL captured output for layer 0.
    npu0 = np.asarray(npu_outputs[0][:SEQ_REAL], dtype=np.float32)
    real0 = real_outputs[0].astype(np.float32)
    cos = float(
        np.dot(npu0.ravel(), real0.ravel())
        / (np.linalg.norm(npu0.ravel()) * np.linalg.norm(real0.ravel()) + 1e-9)
    )
    print(f"\nLayer 0 cosine (NPU-path vs real lerobot backbone): {cos:.6f}")
    print(f"  npu0 mean/std: {npu0.mean():.4f}/{npu0.std():.4f}  real0 mean/std: {real0.mean():.4f}/{real0.std():.4f}")

    if args.profile:
        print(f"\nProfiling {args.layers} layers, {args.reps} reps...")
        x = pad_seq(layer0["hidden_in"].astype(bfloat16), SEQ_PAD)
        cache.profiler.kernel_breakdowns.clear()
        cache.profiler.cpu_times.clear()
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            xx = x
            for i in range(args.layers):
                xx = run_one_layer(xx, i)
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        print(f"{args.layers}-layer NPU (cpu_attn=True) wall: median {times[len(times)//2]:.2f} ms, "
              f"min {times[0]:.2f}, max {times[-1]:.2f}")
        print("CPU baseline (Vu_exp/smolvla_backbone_perf/BACKBONE_PROFILE.md): full fill 43.9 ms")

        print("\n--- Per-ELF breakdown (avg per invocation, all reps) ---")
        for name, entries in sorted(cache.profiler.kernel_breakdowns.items()):
            n = len(entries)
            avg_w = sum(e["write_ms"] for e in entries) / n
            avg_k = sum(e["kernel_ms"] for e in entries) / n
            avg_r = sum(e["read_ms"] for e in entries) / n
            print(f"  {name:20s} write={avg_w:7.3f}ms  device={avg_k:7.3f}ms  read={avg_r:7.3f}ms"
                  f"  total={avg_w+avg_k+avg_r:7.3f}ms  (x{n} calls, {n // args.layers // args.reps if n else 0}/layer)")
        if cache.profiler.cpu_times:
            print("\n--- CPU-side ops (attention fallback etc.) ---")
            for name, ts in sorted(cache.profiler.cpu_times.items()):
                print(f"  {name:20s} avg={sum(ts)/len(ts)*1000:7.3f}ms  (x{len(ts)})")


if __name__ == "__main__":
    main()
