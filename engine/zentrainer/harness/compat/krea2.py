"""Krea 2 training keeps q/k/v in native (B, L, H, D) layout through q/k norms and RoPE (avoiding
contiguous B H L D copies before FlashAttention) and passes compact key/value heads to ComfyUI's
grouped-query attention. With memory.fused_qk_rope, the q/k norms and rotary embedding run as one
trainable op.
"""

from __future__ import annotations

import torch

from .. import compat


def _apply_rope_blhd(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE on (B, L, H, D) with freqs (B or 1, 1, L_f, D/2, 2, 2), matching
    ComfyUI's _apply_rope1 in-place accumulation to avoid materializing two fp32 products."""
    f = freqs.transpose(1, 2)
    if f.shape[1] != 1 and f.shape[1] != x.shape[1]:
        f = f[:, :x.shape[1]]
    x_ = x.to(dtype=f.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    x_out = f[..., 0] * x_[..., 0]
    x_out.addcmul_(f[..., 1], x_[..., 1])
    return x_out.reshape(x.shape).type_as(x)


def apply() -> None:
    if "krea2" in compat._applied:
        return
    try:
        import comfy.ldm.krea2.model as k2
    except Exception:  # an older ComfyUI without Krea 2
        return
    import comfy.model_management as mm

    attention = getattr(k2, "Attention", None)
    if attention is None:
        return
    sampling_forward = attention.forward

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        if not (compat.in_run and mm.in_training):
            return sampling_forward(self, x, freqs, mask, transformer_options)
        patches = transformer_options.get("patches", {})
        has_attn1 = "block_index" in transformer_options and "attn1_patch" in patches
        norms = self.qknorm
        use_fused = (compat.fused_qk_rope and freqs is not None and not has_attn1
                     and not norms.qnorm.scale.requires_grad and not norms.knorm.scale.requires_grad)

        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        # Keep (B, L, H, D) memory layout when there is no attn1_patch (which expects B H L D) and
        # L > 1 (so freqs.transpose(1, 2) is unambiguous for comfy_kitchen's layout detection).
        # Transposing right before attention yields zero-copy views in FlashAttention's native strides.
        if not has_attn1 and q.shape[1] > 1:
            B, L, _ = q.shape
            q = q.reshape(B, L, self.heads, -1)
            k = k.reshape(B, L, self.kvheads, -1)
            v = v.reshape(B, L, self.kvheads, -1)
            if use_fused:
                from ..kernels.qk_norm_rope import qk_norm_rope

                wq = mm.cast_to(norms.qnorm.scale, dtype=torch.float32, device=x.device) + 1.0
                wk = mm.cast_to(norms.knorm.scale, dtype=torch.float32, device=x.device) + 1.0
                q, k = qk_norm_rope(q, k, wq, wk, freqs.transpose(1, 2), norms.qnorm.eps, seq_first=True)
            else:
                q, k = norms(q, k)
                if freqs is not None:
                    q, k = _apply_rope_blhd(q, freqs), _apply_rope_blhd(k, freqs)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        else:
            q = k2.rearrange(q, "B L (H D) -> B H L D", H=self.heads)
            k = k2.rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
            v = k2.rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
            if use_fused:
                from ..kernels.qk_norm_rope import qk_norm_rope

                # the norms' (1 + scale) weights in fp32, as ComfyUI's RMSNorm builds them
                wq = mm.cast_to(norms.qnorm.scale, dtype=torch.float32, device=x.device) + 1.0
                wk = mm.cast_to(norms.knorm.scale, dtype=torch.float32, device=x.device) + 1.0
                q, k = qk_norm_rope(q, k, wq, wk, freqs, norms.qnorm.eps)
            else:
                q, k = norms(q, k)
                if has_attn1:
                    extra_options = transformer_options.copy()
                    for p in patches["attn1_patch"]:
                        out = p(q, k, v, pe=freqs, attn_mask=mask, extra_options=extra_options)
                        q, k, v = out.get("q", q), out.get("k", k), out.get("v", v)
                        freqs, mask = out.get("pe", freqs), out.get("attn_mask", mask)
                if freqs is not None:
                    q, k = k2.apply_rope(q, k, freqs)

        gqa = k.shape[1] != q.shape[1]
        q, k, v = k2.AttentionTensorContainer(q), k2.AttentionTensorContainer(k), k2.AttentionTensorContainer(v)
        out = k2.optimized_attention_masked(q, k, v, self.heads, mask=mask, skip_reshape=True,
                                            preferred_attention=self.comfy_attention,
                                            transformer_options=transformer_options, enable_gqa=gqa)
        if "block_index" in transformer_options and "attn1_output_patch" in patches:
            extra_options = transformer_options.copy()
            for p in patches["attn1_output_patch"]:
                out = p(out, extra_options)
        return self.wo(out * k2.F.sigmoid(gate))

    attention.forward = forward
    compat._applied.add("krea2")
