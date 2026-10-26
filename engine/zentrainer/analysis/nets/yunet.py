"""YuNet, OpenCV's face detector (opencv_zoo face_detection_yunet_2023mar, MIT): depthwise-separable
convolutions throughout, with its batch norms folded into them as the exported file has them."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class _DP(nn.Module):
    """A 1x1 conv, then a depthwise 3x3 (+ relu, except in the heads)."""

    def __init__(self, cin: int, cout: int, relu: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, groups=cout)
        self.relu = relu

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return F.relu(x) if self.relu else x


class _Stage(nn.Module):
    def __init__(self, cin: int, mid: int, cout: int):
        super().__init__()
        self.conv1, self.conv2 = _DP(cin, mid), _DP(mid, cout)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class YuNet(nn.Module):
    """(1, 3, H, W) BGR in 0-255 (H, W multiples of 32) -> per stride (8, 16, 32): (score logit,
    objectness logit, box (dx, dy, log w, log h), 5 keypoints (dx, dy) each), all in stride units
    from the cell's corner, one prediction a cell, rows of the grid in order."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, 2, 1)
        self.stem_dp = _DP(16, 16)
        self.stages = nn.ModuleList([_Stage(16, 16, 32), _Stage(32, 32, 64), _Stage(64, 64, 64),
                                     _Stage(64, 64, 64), _Stage(64, 64, 64)])
        self.lateral = nn.ModuleList([_DP(64, 64) for _ in range(3)])  # strides 8, 16, 32
        self.heads = nn.ModuleDict({k: nn.ModuleList([_DP(64, n, relu=False) for _ in range(3)])
                                    for k, n in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10))})

    def forward(self, x):
        x = F.max_pool2d(self.stem_dp(F.relu(self.stem(x))), 2)
        x = self.stages[1](self.stages[0](x))
        c8 = self.stages[2](F.max_pool2d(x, 2))
        c16 = self.stages[3](F.max_pool2d(c8, 2))
        c32 = self.stages[4](F.max_pool2d(c16, 2))
        p32 = self.lateral[2](c32)
        p16 = self.lateral[1](c16 + F.interpolate(p32, scale_factor=2, mode="nearest"))
        p8 = self.lateral[0](c8 + F.interpolate(p16, scale_factor=2, mode="nearest"))
        flat = lambda t: t.permute(0, 2, 3, 1).reshape(t.shape[0], -1, t.shape[1])  # noqa: E731
        return [tuple(flat(self.heads[k][i](p)) for k in ("cls", "obj", "bbox", "kps")) for i, p in enumerate((p8, p16, p32))]
