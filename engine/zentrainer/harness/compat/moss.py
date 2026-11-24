"""MOSS-Image (comfy/ldm/moss_image/model.py):
- `modulated_rms_norm` and `gated_residual`: ComfyUI's training branches drop the cast back to
  `x.dtype` that its fused inference ops (`ck.rms_adaln`, in-place `x.addcmul_`) do, leaking fp32
  into the quantized `qkv` / `ffn_up_gate` layers and the residual stream (breaking fused bypass
  adapters, doubling saved activation memory, and forcing a second compiled graph after block 0).
  Cast back to `x.dtype`.
- Attention (with memory.fused_qk_rope): the q/k RMSNorms and the rope run as one op that trains
  (kernels/qk_norm_rope.py) instead of a norm, a cast and a rope each. Falls back to ComfyUI's for
  attention patches, a norm weight that trains (or is patched), or no rope. Its sampling already
  fuses them.
"""

from __future__ import annotations

import torch

from .. import compat


def apply() -> None:
    if "moss" in compat._applied:
        return
    try:
        import comfy.ldm.moss_image.model as moss
    except Exception:  # an older ComfyUI without MOSS-Image
        return
    import comfy.model_management as mm

    sampling_attention = getattr(moss, "attention", None)
    if sampling_attention is None:
        return

    sampling_modulated_rms_norm = getattr(moss, "modulated_rms_norm", None)
    if sampling_modulated_rms_norm is not None:
        def modulated_rms_norm(x, norm, scale, shift):
            if not (compat.in_run and mm.in_training):
                return sampling_modulated_rms_norm(x, norm, scale, shift)
            return (norm(x) * (1 + scale) + shift).to(x.dtype)

        moss.modulated_rms_norm = modulated_rms_norm

    sampling_gated_residual = getattr(moss, "gated_residual", None)
    if sampling_gated_residual is not None:
        def gated_residual(x, y, gate):
            if not (compat.in_run and mm.in_training):
                return sampling_gated_residual(x, y, gate)
            return torch.addcmul(x, y, gate).to(x.dtype)

        moss.gated_residual = gated_residual

    def plain(norm) -> bool:
        """A norm whose weight is a frozen tensor with nothing patched onto it."""
        w = getattr(norm, "weight", None)
        return isinstance(w, torch.Tensor) and not w.requires_grad and not getattr(norm, "weight_function", None)

    # Newer ComfyUI passes more (ref_kv, kv_out, mask: MOSS-Image edit, #17296 #17320), positionally
    # or by name; they go through untouched, and the fused path takes only calls that use none.
    def attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options, *more, **named):
        if not (compat.fused_qk_rope and mm.in_training) or not (plain(q_norm) and plain(k_norm)) \
                or any(x is not None for x in (*more, *named.values())):
            return sampling_attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options,
                                      *more, **named)
        from ..kernels.qk_norm_rope import qk_norm_rope

        B, N, _ = q.shape
        wq = mm.cast_to(q_norm.weight, dtype=torch.float32, device=q.device)
        wk = mm.cast_to(k_norm.weight, dtype=torch.float32, device=k.device)
        q, k = qk_norm_rope(q.view(B, N, heads, -1), k.view(B, N, heads, -1), wq, wk, pe, q_norm.eps,
                            split_half=True, seq_first=True)
        q, k, v = (moss.AttentionTensorContainer(q.flatten(2)), moss.AttentionTensorContainer(k.flatten(2)),
                   moss.AttentionTensorContainer(v))
        return moss.optimized_attention(q, k, v, heads, transformer_options=transformer_options,
                                        preferred_attention=comfy_attention)

    moss.attention = attention
    compat._applied.add("moss")
