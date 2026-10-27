"""The raw CUDA W8A8 input gradient (harness/kernels/w8a8_cuda.py) against the exact product."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None
                          or torch.cuda.get_device_capability() < (8, 0), reason="needs an NVIDIA card, sm_80+")


def _regular_hadamard(size: int) -> torch.Tensor:  # ConvRot's: H4 x H4 x ..., normalised
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / size ** 0.5


@cuda
@pytest.mark.parametrize("m,n,k", [(300, 128, 256), (1000, 1536, 768)])  # rows not a multiple of the tile
def test_quantized_product_matches_the_exact_one_rotated(m, n, k):
    from zentrainer.harness.kernels import w8a8_cuda

    torch.manual_seed(0)
    g = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    s = torch.rand(n, device="cuda") * 0.01 + 0.001
    q = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    assert w8a8_cuda.supported(g, q, 256)
    seed = torch.tensor([7], device="cuda", dtype=torch.int32)
    ks = w8a8_cuda._load(g.device)
    gq = torch.empty((m, n), device="cuda", dtype=torch.int8)
    steps = torch.empty(m, device="cuda")
    ks["zt_quantize_rows"](grid=(m, 1, 1), block=(256, 1, 1), args=[g, s, gq, steps, seed, n, g.stride(0)])
    # stored position 4t + j of each 16 holds column 2t + (j & 1) + 8 (j >> 1)
    perm = [2 * (p >> 2) + (p & 1) + 8 * ((p & 3) >> 1) for p in range(16)]
    where = torch.tensor([perm.index(c) for c in range(16)], device="cuda")
    gq = gq[:, (torch.arange(n, device="cuda") // 16 * 16) + where.repeat(n // 16)]

    gs = g.double() * s.double()
    assert torch.allclose(steps.double(), gs.abs().amax(1) / 127, rtol=1e-5)
    err = gq.double() * steps.double()[:, None] - gs
    assert (err.abs() <= steps.double()[:, None] * 1.0001).all()  # a neighbouring int8 step
    assert abs(err.mean().item()) < 0.01 * steps.double().mean().item()  # stochastic: unbiased

    exact = (gq.double() @ q.double()) * steps.double()[:, None]
    rotated = (exact.view(m, k // 256, 256) @ _regular_hadamard(256).cuda().T).view(m, k)
    for rotate, want in ((True, rotated), (False, exact)):
        out = w8a8_cuda.input_grad(g, q, s, seed, rotate).double()
        assert ((out - want).norm() / want.norm()).item() < 3e-3  # one bf16 rounding (~1.7e-3)


def test_shapes_it_does_not_take_go_elsewhere():
    from zentrainer.harness.kernels import w8a8_cuda

    g = torch.zeros(64, 100, dtype=torch.bfloat16)  # on the CPU, and 100 isn't a multiple of 64
    q = torch.zeros(100, 256, dtype=torch.int8)
    assert not w8a8_cuda.supported(g, q, 256)
    assert not w8a8_cuda.supported(torch.zeros(64, 128, dtype=torch.bfloat16), torch.zeros(128, 256, dtype=torch.int8), 64)
