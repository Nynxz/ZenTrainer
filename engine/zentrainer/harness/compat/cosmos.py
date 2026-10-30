"""Cosmos Predict2 (comfy/ldm/cosmos/predict2.py), and Anima, which is built on it:
- Attention: self-attention's q/k RMSNorms and split-half rope run as comfy_kitchen's fused
  rms_rope_split_half, which has no backward ("Trying to backward through
  comfy_kitchen.rms_rope_split_half.default but no autograd formula was registered"): the first
  step of every run failed. In training they run as kernels/qk_norm_rope.py's op, which trains,
  or plain torch when a norm's weight trains too. Its rope is (1, L, 1, D/2, 2, 2) for
  (B, L, heads, D) queries, as MOSS's. Cross-attention (no rope) is ComfyUI's own.
"""

from __future__ import annotations

import torch

from .. import compat


def apply() -> None:
    if "cosmos" in compat._applied:
        return
    try:
        import comfy.ldm.cosmos.predict2 as predict2
    except Exception:  # an older ComfyUI without Cosmos Predict2
        return
    import comfy.model_management as mm

    attention = getattr(predict2, "Attention", None)
    if attention is None or not hasattr(attention, "compute_qkv"):
        return
    sampling_compute_qkv = attention.compute_qkv

    def compute_qkv(self, x, context=None, rope_emb=None, transformer_options={}):
        if not mm.in_training or not self.is_selfattn or rope_emb is None:
            return sampling_compute_qkv(self, x, context, rope_emb=rope_emb, transformer_options=transformer_options)
        from ..kernels.qk_norm_rope import qk_norm_rope, reference

        q_input = k_input = v_input = x
        patches = transformer_options.get("patches", {}).get("attn1_patch", [])
        if patches:
            extra_options = {**transformer_options, "n_heads": self.n_heads, "dim_head": self.head_dim}
            for patch in patches:
                out = patch(q_input, k_input, v_input, pe=rope_emb, attn_mask=None, extra_options=extra_options)
                q_input, k_input, v_input = out.get("q", q_input), out.get("k", k_input), out.get("v", v_input)
                rope_emb = out.get("pe", rope_emb)
        B, L = x.shape[:2]
        q = self.q_proj(q_input).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(k_input).view(B, L, self.n_heads, self.head_dim)
        v = self.v_norm(self.v_proj(v_input).view(B, L, self.n_heads, self.head_dim))
        wq = mm.cast_to(self.q_norm.weight, dtype=torch.float32, device=q.device)
        wk = mm.cast_to(self.k_norm.weight, dtype=torch.float32, device=k.device)
        eps = self.q_norm.eps
        if wq.requires_grad or wk.requires_grad:  # the norms train: torch, which takes their gradient
            return (reference(q, wq, rope_emb, eps, split_half=True, seq_first=True),
                    reference(k, wk, rope_emb, eps, split_half=True, seq_first=True), v)
        q, k = qk_norm_rope(q, k, wq, wk, rope_emb, eps, split_half=True, seq_first=True)
        return q, k, v

    attention.compute_qkv = compute_qkv
    compat._applied.add("cosmos")
