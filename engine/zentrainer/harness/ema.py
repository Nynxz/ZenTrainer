"""Weight averaging (EMA) of the trained adapter (optim.ema): checkpoints, samples and the saved
LoRA come from a running average of the recent steps rather than whichever step the save landed
on, so each reflects the whole dataset instead of the tail of the shuffle. Fizgig ships 0.98 on
by default and measured it higher and steadier on likeness than off.

The average lives beside the live weights in fp32 (an adapter is small), with a warmup so early
checkpoints don't lag behind the start. The resume state keeps both: the live weights to train
on, the average to keep averaging.
"""

from __future__ import annotations

import contextlib

import torch


class EMA:
    def __init__(self, params: dict[str, torch.nn.Parameter], decay: float):
        self.params = params
        self.decay = decay
        self.shadow = {k: p.detach().float().clone() for k, p in params.items()}

    def rate(self, step: int) -> float:
        return min(self.decay, (1.0 + step) / (10.0 + step))  # short warmup: early saves track the weights

    @torch.no_grad()
    def update(self, step: int) -> None:
        w = 1.0 - self.rate(step)
        # group by device and whether params are already fp32 so the update is one multi-tensor kernel
        fp32: dict[torch.device, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        other: dict[torch.device, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for k, p in self.params.items():
            s = self.shadow[k]
            dst = fp32 if p.dtype == torch.float32 else other
            ss, ps = dst.setdefault(s.device, ([], []))
            ss.append(s)
            ps.append(p.detach())
        for ss, ps in fp32.values():
            torch._foreach_lerp_(ss, ps, w)
        for ss, ps in other.values():
            torch._foreach_lerp_(ss, [p.float() for p in ps], w)

    @contextlib.contextmanager
    def applied(self):
        """The averaged weights in place for a moment (a save, a sample); yields the live ones."""
        live = {k: p.detach().clone() for k, p in self.params.items()}
        try:
            with torch.no_grad():
                for k, p in self.params.items():
                    p.copy_(self.shadow[k].to(p.dtype))
            yield live
        finally:
            with torch.no_grad():
                for k, p in self.params.items():
                    p.copy_(live[k])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.get("shadow", {}).items():
            if k in self.shadow and self.shadow[k].shape == v.shape:
                self.shadow[k].copy_(v.to(self.shadow[k]))
