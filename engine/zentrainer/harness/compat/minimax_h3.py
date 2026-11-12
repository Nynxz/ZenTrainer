"""MiniMax H3 (comfy/ldm/minimax/model.py):
- Attention: the training branch calls comfy_kitchen's fused RMSNorm + rope, which has no
  backward. Plain torch here: the q/k RMSNorm modules, then the partial split-half rope (the
  first `rot` of each head's dims rotate, the rest pass through). Matches the kitchen op to ~1e-7.
  With memory.fused_qk_rope, norms and rope run as one op that trains (kernels/qk_norm_rope.py).
- `_mod_gate` adds the gated branch into the residual stream in place, but norm1/norm2 saved
  that stream for their backward ("modified by an inplace operation"). Out of place in training.
- MiniMax-H3-Image has an output head per step of its own sigma grid, picked by where the step
  sits in the sampler's schedule (transformer_options["sample_sigmas"]). Training has no
  schedule, so the grid stands in: a σ on the grid trains that one head.
"""

from __future__ import annotations

import logging

import torch

from .. import compat


def apply() -> None:
    if "minimax_h3" in compat._applied:
        return
    try:
        import comfy.ldm.minimax.model as h3
    except Exception:  # an older ComfyUI without H3
        return
    import comfy.model_management as mm

    attention = getattr(h3, "Attention", None)
    gate = getattr(h3, "_mod_gate", None)
    if attention is None or gate is None:
        return
    sampling_forward = attention.forward

    def forward(self, x, rope_freqs=None, transformer_options={}):
        if not mm.in_training or rope_freqs is None:
            return sampling_forward(self, x, rope_freqs, transformer_options)
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        rot = rope_freqs.shape[-3] * 2
        q, k = q.view(1, s, self.heads, self.head_dim), k.view(1, s, self.heads, self.head_dim)
        if compat.fused_qk_rope and _plain(self.q_norm) and _plain(self.k_norm):
            from ..kernels.qk_norm_rope import qk_norm_rope

            wq = mm.cast_to(self.q_norm.weight, dtype=torch.float32, device=q.device)
            wk = mm.cast_to(self.k_norm.weight, dtype=torch.float32, device=k.device)
            q, k = qk_norm_rope(q, k, wq, wk, rope_freqs, self.q_norm.eps, split_half=True, seq_first=True, rot=rot)
            q, k = q[0], k[0]
        else:
            q = _rope_split_half(self.q_norm(q), rope_freqs, rot)[0]
            k = _rope_split_half(self.k_norm(k), rope_freqs, rot)[0]
        q = h3.AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k = h3.AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v = h3.AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        out = h3.optimized_attention(q, k, v, self.heads, preferred_attention=self.comfy_attention, mask=None,
                                     skip_reshape=True, transformer_options=transformer_options)
        return self.out_proj(out.squeeze(0))

    def mod_gate(x, g, other, segments):
        if not mm.in_training:
            return gate(x, g, other, segments)
        parts, end = [], 0
        for a, b, row in segments:
            if a > end:
                parts.append(x[end:a])
            parts.append(torch.addcmul(x[a:b], other[a:b], h3._mod_row(g, row, x.dtype)))
            end = b
        if end < x.shape[0]:
            parts.append(x[end:])
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    attention.forward = forward
    h3._mod_gate = mod_gate
    image = getattr(h3, "MiniMaxH3ImageModel", None)
    if image is not None:
        image_forward = image._forward

        def image_forward_grid(self, x, timestep, context, transformer_options={}, **kwargs):
            if mm.in_training and "sample_sigmas" not in transformer_options:
                transformer_options = {**transformer_options, "sample_sigmas": torch.tensor(self.pdd_sigmas)}
            return image_forward(self, x, timestep, context, transformer_options, **kwargs)

        image._forward = image_forward_grid
    compat._applied.add("minimax_h3")
    logging.info("[ZenTrainer] MiniMax H3: trainable attention and residual gate patched in")


def _plain(norm) -> bool:
    """A norm whose weight is a frozen tensor with nothing patched onto it (else the separate ops)."""
    w = getattr(norm, "weight", None)
    return isinstance(w, torch.Tensor) and not w.requires_grad and not getattr(norm, "weight_function", None)


def _rope_split_half(x: torch.Tensor, pe: torch.Tensor, rot: int) -> torch.Tensor:
    """Rotate the first `rot` dims of each head (split-half pairs); pe is (1, S, 1, rot/2, 2, 2)."""
    xr, xp = x[..., :rot], x[..., rot:]
    x_ = xr.to(pe.dtype).unflatten(-1, (2, -1)).movedim(-2, -1).unsqueeze(-2)
    out = (pe[..., 0] * x_[..., 0] + pe[..., 1] * x_[..., 1]).movedim(-1, -2).flatten(-2).type_as(x)
    return torch.cat([out, xp], -1)
