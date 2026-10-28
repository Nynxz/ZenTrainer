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


def _quantized(g, s, seed):
    """The quantizer's int8 rows in column order, and each row's step."""
    from zentrainer.harness.kernels import w8a8_cuda

    m, n = g.shape
    ks = w8a8_cuda._load(g.device)
    gq = torch.empty((m, n), device="cuda", dtype=torch.int8)
    steps = torch.empty(m, device="cuda")
    ks["zt_quantize_rows"](grid=(m, 1, 1), block=(256, 1, 1), args=[g, s, gq, steps, seed, n, g.stride(0)])
    # stored position 4t + j of each 16 holds column 2t + (j & 1) + 8 (j >> 1)
    perm = [2 * (p >> 2) + (p & 1) + 8 * ((p & 3) >> 1) for p in range(16)]
    where = torch.tensor([perm.index(c) for c in range(16)], device="cuda")
    return gq[:, (torch.arange(n, device="cuda") // 16 * 16) + where.repeat(n // 16)], steps


# rows: one, under and over a tile (128), odd; the layer shapes of Krea 2 and MOSS (out, in)
@cuda
@pytest.mark.parametrize("m,n,k", [(1, 64, 256), (17, 128, 256), (127, 128, 512), (129, 192, 256), (300, 128, 256),
                                   (1000, 1536, 768), (4097, 1536, 6144), (512, 6144, 1536), (512, 16384, 6144),
                                   (512, 6144, 16384), (512, 9216, 3072), (512, 3072, 8192)])
def test_quantized_product_matches_the_exact_one_rotated(m, n, k):
    from zentrainer.harness.kernels import w8a8_cuda

    torch.manual_seed(0)
    g = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    s = torch.rand(n, device="cuda") * 0.01 + 0.001
    q = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    assert w8a8_cuda.supported(g, q, 256)
    seed = torch.tensor([7], device="cuda", dtype=torch.int32)
    gq, steps = _quantized(g, s, seed)

    gs = g.double() * s.double()
    assert torch.allclose(steps.double(), gs.abs().amax(1) / 127, rtol=1e-5)
    err = gq.double() * steps.double()[:, None] - gs
    assert (err.abs() <= steps.double()[:, None] * 1.0001).all()  # a neighbouring int8 step
    if m * n >= 100_000:  # one draw's mean error: ~step / sqrt(12 m n); see the seeds test for the bias itself
        assert abs(err.mean().item()) < 0.01 * steps.double().mean().item()

    exact = (gq.double() @ q.double()) * steps.double()[:, None]
    rotated = (exact.view(m, k // 256, 256) @ _regular_hadamard(256).cuda().T).view(m, k)
    for rotate, want in ((True, rotated), (False, exact)):
        out = w8a8_cuda.input_grad(g, q, s, seed, rotate).double()
        assert ((out - want).norm() / want.norm()).item() < 3e-3  # one bf16 rounding (~1.7e-3)
    r = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * want.std().item()  # the LoRA's, added in
    out = w8a8_cuda.input_grad(g, q, s, seed, True, r).double()
    want = rotated + r.double()
    assert ((out - want).norm() / want.norm()).item() < 3e-3


def test_shapes_it_does_not_take_go_elsewhere():
    from zentrainer.harness.kernels import w8a8_cuda

    g = torch.zeros(64, 100, dtype=torch.bfloat16)  # on the CPU, and 100 isn't a multiple of 64
    q = torch.zeros(100, 256, dtype=torch.int8)
    assert not w8a8_cuda.supported(g, q, 256)
    assert not w8a8_cuda.supported(torch.zeros(64, 128, dtype=torch.bfloat16), torch.zeros(128, 256, dtype=torch.int8), 64)


@cuda
def test_a_strided_gradient_and_rows_of_zeros():
    from zentrainer.harness.kernels import w8a8_cuda

    torch.manual_seed(1)
    wide = torch.randn(200, 384, device="cuda", dtype=torch.bfloat16)
    g = wide[:, :256]  # rows 384 apart, as a slice of a bigger gradient
    g[5].zero_()  # a token the loss doesn't reach: step 1e-30 / 127, all zeros, no NaN
    s = torch.rand(256, device="cuda") + 0.5
    q = torch.randint(-127, 128, (256, 512), device="cuda", dtype=torch.int8)
    seed = torch.tensor([3], device="cuda", dtype=torch.int32)
    assert w8a8_cuda.supported(g, q, 256)
    out = w8a8_cuda.input_grad(g, q, s, seed, True).double()
    want = w8a8_cuda.input_grad(g.contiguous(), q, s, seed, True).double()
    assert torch.equal(out, want) and torch.isfinite(out).all() and not out[5].any()


@cuda
def test_the_rounding_is_unbiased_over_seeds():
    torch.manual_seed(2)
    g = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    s = torch.rand(256, device="cuda") + 0.5
    gs = g.double() * s.double()
    mean = torch.zeros_like(gs)
    for i in range(400):
        gq, steps = _quantized(g, s, torch.tensor([i], device="cuda", dtype=torch.int32))
        mean += gq.double() * steps.double()[:, None] / 400
    step = (gs.abs().amax(1, keepdim=True) / 127)
    # one draw is off by up to a step; the mean of 400 by ~step/sqrt(400 * 6): well under a tenth
    assert ((mean - gs).abs() / step).max().item() < 0.12
    assert abs(((mean - gs) / step).mean().item()) < 0.005


@cuda
def test_what_it_does_not_take_goes_to_the_triton_path(monkeypatch):
    from zentrainer.harness.kernels import w8a8_cuda

    q = torch.zeros(128, 256, dtype=torch.int8, device="cuda")
    g = torch.zeros(64, 128, dtype=torch.bfloat16, device="cuda")
    assert w8a8_cuda.supported(g, q, 256) and w8a8_cuda.supported(g, q, 0)
    assert not w8a8_cuda.supported(g.half(), q, 256)  # fp16 and fp32 gradients
    assert not w8a8_cuda.supported(g.float(), q, 256)
    assert not w8a8_cuda.supported(g, q, 64)  # another rotation size
    assert not w8a8_cuda.supported(g, torch.zeros(128, 200, dtype=torch.int8, device="cuda"), 256)  # in % 256
    assert not w8a8_cuda.supported(g, q.t().contiguous().t(), 256)  # the weight not as stored
    assert not w8a8_cuda.supported(g[:, 1:65], torch.zeros(64, 256, dtype=torch.int8, device="cuda"), 256)  # unaligned
    monkeypatch.setenv("ZENTRAINER_CUDA_KERNELS", "0")
    monkeypatch.setattr(w8a8_cuda, "_kernels", {})
    assert not w8a8_cuda.supported(g, q, 256)


def test_lora_fused_with_an_int8_layer_has_the_bypass_gradients(monkeypatch):
    # _LoRAInt8 = base(x) + up(down(x)) * scale, the base's int8 parts stood in for by a dense layer
    from zentrainer.harness.kernels import int8_backward as ib

    torch.manual_seed(0)
    base = torch.randn(48, 32, dtype=torch.float64)
    monkeypatch.setattr(ib, "_int8_forward", lambda m, x2, residual: x2 @ base.t() + residual)
    monkeypatch.setattr(ib, "_layer_input_grad", lambda m, g, residual=None: g @ base + residual)
    x = torch.randn(2, 5, 32, dtype=torch.float64, requires_grad=True)
    down = torch.randn(4, 32, dtype=torch.float64, requires_grad=True)
    up = torch.randn(48, 4, dtype=torch.float64, requires_grad=True)
    out = ib._LoRAInt8.apply(x, down, up, 0.75, None, None)
    want = x @ base.t() + (x @ down.t()) @ up.t() * 0.75
    assert torch.allclose(out, want)
    g = torch.randn_like(want)
    got = torch.autograd.grad(out, (x, down, up), g)
    ref = torch.autograd.grad(want, (x, down, up), g)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b)


@cuda
def test_tensors_it_does_not_suit_go_to_the_triton_path_inside_the_op(monkeypatch):
    # a trace decides on the CUDA path before the real tensors exist; one that isn't 16-byte
    # aligned (a view one element in) has to come back by the fallback, still the same product
    from zentrainer.harness.kernels import int8_backward, w8a8_cuda

    # ConvRot's rotation, which the fallback takes from comfy_kitchen (ComfyUI's, not here)
    monkeypatch.setattr(int8_backward, "_hadamard", lambda size, device, dtype: _regular_hadamard(size).to(device, dtype))

    torch.manual_seed(4)
    flat = torch.randn(64 * 256 + 1, device="cuda", dtype=torch.bfloat16)
    g = flat[1:].view(64, 256)  # 2 bytes in: not aligned
    assert g.data_ptr() % 16
    s = torch.rand(256, device="cuda") * 0.01 + 0.001
    q = torch.randint(-127, 128, (256, 512), device="cuda", dtype=torch.int8)
    seed = torch.tensor([5], device="cuda", dtype=torch.int32)
    exact = (g.double() * s.double()) @ q.double()
    for rotate in (True, False):
        want = (exact.view(64, 2, 256) @ _regular_hadamard(256).cuda().T).view(64, 512) if rotate else exact
        out = w8a8_cuda.input_grad(g, q, s, seed, rotate)
        assert out.dtype == torch.bfloat16 and out.shape == (64, 512)
        assert ((out.double() - want).norm() / want.norm()).item() < 2e-2  # int8 rounding, then bf16
