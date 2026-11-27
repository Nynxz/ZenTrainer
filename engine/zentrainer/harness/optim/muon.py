"""Adapter-native Muon optimizer (Newton-Schulz polar orthogonalized momentum).

Applies 5-step quintic Newton-Schulz polar orthogonalization to Nesterov momentum:
- 2D matrices (LoKr W1/W2, LoRA up/down): orthogonalized and scaled by sqrt(max(m, n)) so entry
  RMS is ~1.0 (matching AdamW's LR scale) while giving uniform singular-value steps across the
  full-rank Kronecker product (U1 kron U2 is orthogonal).
- 3D square OFT blocks (k, b, b): projected to the skew-symmetric Lie algebra so(b) before
  batched polar orthogonalization (Geodesic Skew-Muon).
- 1D / other tensors: RMS-normalized Nesterov momentum fallback.
"""

from __future__ import annotations

import math

import torch


def _ns5_2d(g: torch.Tensor) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    nrm = x.norm()
    if nrm < 1e-12:
        return torch.zeros_like(g, dtype=torch.float32)
    x = x / (nrm + 1e-7)
    for _ in range(5):
        aa = x @ x.T
        x = a * x + (b * aa + c * (aa @ aa)) @ x
    return x.T if transposed else x


def _ns5_batched_skew(g: torch.Tensor) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = 0.5 * (g.float() - g.float().transpose(1, 2))
    nrm = x.flatten(1).norm(dim=1).view(-1, 1, 1)
    x = torch.where(nrm > 1e-12, x / (nrm + 1e-7), torch.zeros_like(x))
    for _ in range(5):
        aa = x @ x.transpose(1, 2)
        x = a * x + (b * aa + c * (aa @ aa)) @ x
    return 0.5 * (x - x.transpose(1, 2))


class Muon(torch.optim.Optimizer):
    """Pure-PyTorch Newton-Schulz orthogonalized momentum for LoKr, LoRA and OFT."""

    def __init__(self, params, lr: float = 3.0e-4, momentum: float = 0.92, weight_decay: float = 0.01):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"])
            beta = float(group["momentum"])
            wd = float(group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data.float()
                if g.norm() < 1e-12:
                    continue
                is_oft = p.ndim == 3 and p.shape[1] == p.shape[2] and p.shape[1] >= 2
                if is_oft:
                    g = 0.5 * (g - g.transpose(1, 2))
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(beta).add_(g)
                nesterov = g.add(buf, alpha=beta)
                if is_oft:
                    u = _ns5_batched_skew(nesterov) * math.sqrt(p.shape[1])
                elif nesterov.ndim == 2 and min(nesterov.shape) >= 2:
                    u = _ns5_2d(nesterov) * math.sqrt(max(nesterov.shape))
                else:
                    u = nesterov / nesterov.pow(2).mean().sqrt().clamp_min(1e-8)
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(u.to(p.dtype), alpha=-lr)
                if is_oft:
                    p.data.copy_(0.5 * (p.data - p.data.transpose(1, 2)))
        return loss
