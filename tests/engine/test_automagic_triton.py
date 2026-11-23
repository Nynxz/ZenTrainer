"""Numerical parity and edge-case tests for fused Triton Automagic3Batched / Automagic4Batched."""

import pytest
import torch

from zentrainer.harness.optim.automagic3 import Automagic3
from zentrainer.harness.optim.automagic3_batched import Automagic3Batched
from zentrainer.harness.optim.automagic4 import Automagic4
from zentrainer.harness.optim.automagic4_batched import Automagic4Batched


def _run_pair(ref_cls, fast_cls, device, dtype, shapes, steps=14, wd=0.01):
    g = torch.Generator(device="cpu").manual_seed(123)
    inits = [torch.randn(s, generator=g, dtype=dtype).to(device) for s in shapes]
    p_ref = [torch.nn.Parameter(t.clone()) for t in inits]
    p_fast = [torch.nn.Parameter(t.clone()) for t in inits]
    o_ref = ref_cls(p_ref, lr=1e-3, weight_decay=wd, fused=False)
    o_fast = fast_cls(p_fast, lr=1e-3, weight_decay=wd, fused=False)
    for step in range(steps):
        # Mix consistent trends and alternating steps so both up and down votes fire
        sign = 1.0 if step < 8 else (-1.0 if step % 2 == 0 else 1.0)
        for pr, pf in zip(p_ref, p_fast):
            gr = (torch.randn(pr.shape, generator=g) * 0.1 + sign * 0.2).to(device=device, dtype=dtype)
            pr.grad = gr.clone()
            pf.grad = gr.clone()
        o_ref.step()
        o_fast.step()
    return p_ref, p_fast, o_ref, o_fast


@pytest.mark.parametrize("pair", [(Automagic3, Automagic3Batched), (Automagic4, Automagic4Batched)])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))])
def test_automagic_batched_matches_vendored_fp32_including_odd_shapes(pair, device):
    ref_cls, fast_cls = pair
    # Include non-multiple-of-8 shapes (7x13 = 91 elems) alongside typical LoRA shapes
    shapes = [(16, 64), (16, 64), (64, 16), (64, 16), (7, 13), (7, 13), (32,)]
    p_ref, p_fast, o_ref, o_fast = _run_pair(ref_cls, fast_cls, device, torch.float32, shapes, steps=14, wd=0.02)
    for pr, pf in zip(p_ref, p_fast):
        torch.testing.assert_close(pf.detach(), pr.detach(), atol=1e-5, rtol=1e-5)
        assert torch.equal(o_fast.state[pf]["sign_history"], o_ref.state[pr]["sign_history"])
        torch.testing.assert_close(o_fast.state[pf]["lr"], o_ref.state[pr]["lr"], atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("pair", [(Automagic3, Automagic3Batched), (Automagic4, Automagic4Batched)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_automagic_triton_stochastic_rounding_moves_small_updates(pair, dtype):
    _, fast_cls = pair
    p = [torch.nn.Parameter(torch.ones((16, 128), device="cuda", dtype=dtype)) for _ in range(4)]
    opt = fast_cls(p, lr=1e-4, weight_decay=0.0, fused=False)
    for _ in range(40):
        for t in p:
            t.grad = torch.ones_like(t) * 0.1
        opt.step()
    # Small updates below 1 ULP must accumulate stochastically instead of stalling at 1.0
    mean_val = sum(t.detach().float().mean().item() for t in p) / len(p)
    assert mean_val < 0.999 and torch.isfinite(p[0]).all()
