"""Fused _LoRAInt8 bypass under _Assisted (training_adapter, turbo_lora, base_lora, slider)."""

import pytest
import torch
import torch.nn.functional as F

from zentrainer.harness.kernels import int8_backward
from zentrainer.harness.network import _Assisted, _Sliced, _Switch


def _fake_lora(in_dim: int, out_dim: int, rank: int, alpha: float, device, dtype, requires_grad: bool):
    from comfy.weight_adapter.lora import LoraDiff

    mat1 = torch.randn(out_dim, rank, device=device, dtype=dtype) * 0.05
    mat2 = torch.randn(rank, in_dim, device=device, dtype=dtype) * 0.05
    mod = LoraDiff((mat1, mat2, alpha, None, None, None)).to(device=device, dtype=dtype)
    mod.requires_grad_(requires_grad)
    if isinstance(getattr(mod, "alpha", None), torch.nn.Parameter):
        mod.alpha.requires_grad_(False)
    return mod


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("case", ["turbo_only", "assist_and_base", "assist_mult", "sliced_assist", "slider_neg"])
def test_assisted_lora_beside_matches_unfused_forward_and_backward(case, monkeypatch):
    comfy_kitchen = pytest.importorskip("comfy_kitchen.tensor")
    import comfy.model_management as mm
    import comfy.ops as ops

    dev = torch.device("cuda")
    torch.manual_seed(7)
    in_dim, out_dim = 256, 256
    layer = ops.mixed_precision_ops({"weight": {"format": "int8_tensorwise"}}).Linear(in_dim, out_dim, bias=False)
    w_f = torch.randn(out_dim, in_dim, device=dev, dtype=torch.bfloat16) * 0.05
    row_scale = w_f.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-6) / 127.0
    q = comfy_kitchen.QuantizedTensor.from_float(w_f, "TensorWiseINT8Layout", scale=row_scale)
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.weight_function = []
    layer.bias_function = []
    layer.layout_type = "TensorWiseINT8Layout"
    layer.quant_format = "int8_tensorwise"

    trained = _fake_lora(in_dim, out_dim, 8, 8.0, dev, torch.float32, True)
    sw = _Switch()
    assist = base = preview = None
    base_strength = 1.0
    mult = 1.0
    if case == "turbo_only":
        preview = _fake_lora(in_dim, out_dim, 4, 4.0, dev, torch.bfloat16, False)
    elif case in ("assist_and_base", "assist_mult"):
        assist = _fake_lora(in_dim, out_dim, 8, 16.0, dev, torch.bfloat16, False)
        base = _fake_lora(in_dim, out_dim, 4, 4.0, dev, torch.bfloat16, False)
        base_strength = 0.5
        if case == "assist_mult":
            mult = 0.5
    elif case == "sliced_assist":
        assist = _Sliced()
        assist.add(0, 128, _fake_lora(in_dim, 128, 4, 8.0, dev, torch.bfloat16, False))
        assist.add(128, 128, _fake_lora(in_dim, 128, 4, 4.0, dev, torch.bfloat16, False))
    elif case == "slider_neg":
        sw.strength = -1.0

    assisted = _Assisted(trained, assist, sw, preview=preview, preview_strength=0.75,
                         base=base, base_strength=base_strength)
    prev_mode = int8_backward._mode
    int8_backward.enable("w8a8")
    try:
        wrapped = int8_backward.lora_beside(assisted, layer)
        assert isinstance(wrapped, int8_backward.LoRABeside)
        assert list(wrapped.named_children()) == [("assisted", assisted)]
        for attr, val in (("multiplier", mult), ("is_conv", False), ("kw_dict", {})):
            setattr(wrapped, attr, val)

        x_ref = torch.randn(2, 32, in_dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
        x_f = x_ref.detach().clone().requires_grad_(True)
        g_out = torch.randn(2, 32, out_dim, device=dev, dtype=torch.bfloat16)

        # Reference: exact dequantized base + assisted.h()
        trained_ref = _fake_lora(in_dim, out_dim, 8, 8.0, dev, torch.float32, True)
        with torch.no_grad():
            trained_ref.lora_down.weight.copy_(trained.lora_down.weight)
            trained_ref.lora_up.weight.copy_(trained.lora_up.weight)
        assisted_ref = _Assisted(trained_ref, assist, sw, preview=preview, preview_strength=0.75,
                                 base=base, base_strength=base_strength)
        for attr, val in (("multiplier", mult), ("is_conv", False), ("kw_dict", {})):
            setattr(assisted_ref, attr, val)
        base_ref = F.linear(x_ref, q.dequantize().to(torch.bfloat16))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_ref = base_ref + assisted_ref.h(x_ref, base_ref).to(torch.bfloat16)
        y_ref.backward(g_out)

        # Fused LoRABeside path
        prev = getattr(mm, "in_training", False)
        mm.in_training = True
        try:
            y_f = wrapped.bypass_forward(lambda t: F.linear(t, q.dequantize().to(torch.bfloat16)), x_f)
            y_f.backward(g_out)
        finally:
            mm.in_training = prev

        rel_y = (y_f.float() - y_ref.float()).norm() / y_ref.float().norm()
        rel_dx = (x_f.grad.float() - x_ref.grad.float()).norm() / x_ref.grad.float().norm()
        rel_da = (trained.lora_down.weight.grad - trained_ref.lora_down.weight.grad).norm() / trained_ref.lora_down.weight.grad.norm()
        rel_db = (trained.lora_up.weight.grad - trained_ref.lora_up.weight.grad).norm() / trained_ref.lora_up.weight.grad.norm()
        assert rel_y < 0.02 and rel_dx < 0.03 and rel_da < 0.01 and rel_db < 0.01

        # Sampling mode (mm.in_training=False, switch.on=False) delegates to preview/turbo_lora cleanly
        if case == "turbo_only":
            sw.on = False
            mm.in_training = False
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                y_s = wrapped.bypass_forward(lambda t: F.linear(t, q.dequantize().to(torch.bfloat16)), x_f)
                b_s = F.linear(x_f, q.dequantize().to(torch.bfloat16))
                y_s_ref = b_s + assisted_ref.h(x_f, b_s).to(torch.bfloat16)
            torch.testing.assert_close(y_s, y_s_ref, atol=1e-2, rtol=1e-2)
    finally:
        int8_backward.enable(prev_mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("in_dim,out_dim,assisted_mode", [(256, 512, False), (512, 256, True)])
def test_lokr_beside_matches_unfused_forward_and_backward(in_dim, out_dim, assisted_mode):
    comfy_kitchen = pytest.importorskip("comfy_kitchen.tensor")
    import comfy.model_management as mm
    import comfy.ops as ops
    from comfy.weight_adapter.lokr import LoKrAdapter

    dev = torch.device("cuda")
    torch.manual_seed(11)
    layer = ops.mixed_precision_ops({"weight": {"format": "int8_tensorwise"}}).Linear(in_dim, out_dim, bias=False)
    w_f = torch.randn(out_dim, in_dim, device=dev, dtype=torch.bfloat16) * 0.05
    row_scale = w_f.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-6) / 127.0
    q = comfy_kitchen.QuantizedTensor.from_float(w_f, "TensorWiseINT8Layout", scale=row_scale)
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.weight_function = []
    layer.bias_function = []
    layer.layout_type = "TensorWiseINT8Layout"
    layer.quant_format = "int8_tensorwise"

    trained = LoKrAdapter.create_train(w_f, rank=4, alpha=1.0).to(device=dev, dtype=torch.float32)
    with torch.no_grad():
        trained.lokr_w1.normal_(0, 0.05)
        trained.lokr_w2.normal_(0, 0.05)
    trained.alpha.requires_grad_(False)

    trained_ref = LoKrAdapter.create_train(w_f, rank=4, alpha=1.0).to(device=dev, dtype=torch.float32)
    with torch.no_grad():
        trained_ref.lokr_w1.copy_(trained.lokr_w1)
        trained_ref.lokr_w2.copy_(trained.lokr_w2)
    trained_ref.alpha.requires_grad_(False)

    mult = 0.75 if assisted_mode else 1.0
    if assisted_mode:
        sw = _Switch()
        sw.strength = 0.5
        assist = _fake_lora(in_dim, out_dim, 4, 8.0, dev, torch.bfloat16, False)
        adapter = _Assisted(trained, assist, sw)
        adapter_ref = _Assisted(trained_ref, assist, sw)
    else:
        adapter, adapter_ref = trained, trained_ref

    prev_mode = int8_backward._mode
    int8_backward.enable("w8a8")
    try:
        wrapped = int8_backward.lora_beside(adapter, layer)
        assert isinstance(wrapped, int8_backward.LoKrBeside)
        for attr, val in (("multiplier", mult), ("is_conv", False), ("kw_dict", {})):
            setattr(wrapped, attr, val)
            setattr(adapter_ref, attr, val)

        x_ref = torch.randn(2, 32, in_dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
        x_f = x_ref.detach().clone().requires_grad_(True)
        g_out = torch.randn(2, 32, out_dim, device=dev, dtype=torch.bfloat16)

        base_ref = F.linear(x_ref, q.dequantize().to(torch.bfloat16))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_ref = base_ref + adapter_ref.h(x_ref, base_ref).to(torch.bfloat16)
        y_ref.backward(g_out)

        prev = getattr(mm, "in_training", False)
        mm.in_training = True
        try:
            y_f = wrapped.bypass_forward(lambda t: F.linear(t, q.dequantize().to(torch.bfloat16)), x_f)
            y_f.backward(g_out)
        finally:
            mm.in_training = prev

        rel_y = (y_f.float() - y_ref.float()).norm() / y_ref.float().norm()
        rel_dx = (x_f.grad.float() - x_ref.grad.float()).norm() / x_ref.grad.float().norm()
        rel_dw1 = (trained.lokr_w1.grad - trained_ref.lokr_w1.grad).norm() / trained_ref.lokr_w1.grad.norm()
        rel_dw2 = (trained.lokr_w2.grad - trained_ref.lokr_w2.grad).norm() / trained_ref.lokr_w2.grad.norm()
        assert rel_y < 0.02 and rel_dx < 0.03 and rel_dw1 < 0.01 and rel_dw2 < 0.01
    finally:
        int8_backward.enable(prev_mode)
