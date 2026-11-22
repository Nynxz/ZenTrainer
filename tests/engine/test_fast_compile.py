"""Tests for shared-graph block compilation and pre-compile eager keep-auto probing."""

import sys
import types

import torch
import torch.nn as nn

from zentrainer.harness import trainer as T
from zentrainer.harness.kernels.compile import compiled_blocks
from zentrainer.harness.kernels import int8_backward
from zentrainer.harness.memory import gradient_checkpointing


def test_compiled_blocks_and_patch_saving_share_one_wrapper_per_class(monkeypatch):
    nodes_train = types.ModuleType("comfy_extras.nodes_train")

    def fake_patch(m, offloading=False):
        m.org_forward = m.forward
        m.forward = lambda *a, _f=m.forward, **kw: _f(*a, **kw)

    def fake_unpatch(m):
        if hasattr(m, "org_forward"):
            m.forward = m.org_forward
            delattr(m, "org_forward")

    nodes_train.patch = fake_patch
    nodes_train.unpatch = fake_unpatch
    pkg = types.ModuleType("comfy_extras")
    pkg.nodes_train = nodes_train
    monkeypatch.setitem(sys.modules, "comfy_extras", pkg)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_train", nodes_train)

    class TextBlock(nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.l1 = nn.Linear(d, d)
            self.l2 = nn.Linear(d, d)

        def forward(self, x):
            return self.l2(self.l1(x))

    class MainBlock(nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.l1 = nn.Linear(d, d)
            self.l2 = nn.Linear(d, d)

        def forward(self, x):
            return x + self.l2(torch.relu(self.l1(x)))

    class FakeMOSS(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_blocks = nn.ModuleList([TextBlock() for _ in range(2)])
            self.layers = nn.ModuleList([MainBlock() for _ in range(6)])

    dm = FakeMOSS()
    with compiled_blocks(dm, True) as n_comp:
        assert n_comp == 6
        # All 6 MainBlocks share the exact same underlying compiled function object
        funcs = {b.forward.__func__ for b in dm.layers}
        assert len(funcs) == 1
        with gradient_checkpointing(dm, True, depth=1, offloading=False, skip=0, save="attention") as ck:
            # MainBlocks must come BEFORE TextBlocks in keep order so probe never hits 0.00 GB text blocks
            first_six = [ck.all[i] for i in ck.order[:6]]
            assert all(isinstance(b, MainBlock) for b in first_six)
            # All 6 checkpointed MainBlocks share one compiled selective-checkpoint wrapper
            ckpt_funcs = {b.forward.__func__ for b in dm.layers}
            assert len(ckpt_funcs) == 1
            # Un-checkpointing restores the shared un-checkpointed compiled method cleanly
            ck.keep(6)
            assert {b.forward.__func__ for b in dm.layers} == funcs
    # All wrappers cleanly removed on exit
    assert all("forward" not in b.__dict__ for b in dm.layers)


def test_int8_install_uses_shared_method_object(monkeypatch):
    class FakeQuantLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_format = "int8_tensorwise"
            self.layout_type = "TensorWiseINT8Layout"
            self.weight_function = []
            self.bias_function = []
            self.weight = type("QuantizedTensor", (), {})()

        def forward_comfy_cast_weights(self, x):
            return x

        def forward(self, x):
            return x

    model = nn.Sequential(FakeQuantLinear(), FakeQuantLinear(), FakeQuantLinear())
    n = int8_backward.install(model)
    try:
        assert n == 3
        assert len({m.forward.__func__ for m in model}) == 1
    finally:
        int8_backward.uninstall()
    assert all("forward" not in m.__dict__ and "_zt_orig_forward" not in m.__dict__ for m in model)


def test_precompile_eager_probe_jumps_in_one_shot_and_unhooks_before_compile(monkeypatch):
    stances = []
    monkeypatch.setattr(T.torch.compiler, "set_stance", lambda s: stances.append(s))
    monkeypatch.setattr(T.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(T.torch.cuda, "reset_peak_memory_stats", lambda: None)

    t = T.Trainer.__new__(T.Trainer)
    t._keep_auto = True
    t._compiled = 42
    t._keep_warm = 0
    t._res_seen = 768
    t._peak = 7 * 1024 ** 3
    t._card = lambda: (25 * 1024 ** 3, 0)
    # Step 1 ran a 48x48 bucket; dataset also has a 64x48 bucket -> bucket_scale = 64/48 = 1.333x
    t._step_key = (((1, 16, 48, 48), "None"),)
    t.dataset = types.SimpleNamespace(buckets={(16, 48, 48): [0], (16, 64, 48): [1]})

    blocks = [nn.Linear(4, 4) for _ in range(44)]
    keep_calls = []

    def fake_keep(n):
        t._ckpt.kept = n
        keep_calls.append(n)
        return n

    t._ckpt = types.SimpleNamespace(all=blocks, kept=0, keep=fake_keep)
    t._arm_eager_keep_probe()
    assert t._keep_settle == 1 and t._ckpt.kept == 1 and stances == ["force_eager"]
    assert t._keep_hooks is not None

    # Step 1 (eager): measures 0.30 GB on the single real probe block -> jumps 1 -> 37 and stays eager to verify
    t._keep_cost = {43: int(0.30 * 1024 ** 3)}
    t._keep_step(768)
    assert t._keep_settle == 2
    assert t._ckpt.kept == 37
    assert stances == ["force_eager"]  # still eager for the verification & top-up step!

    # Step 2 (still eager): runs at 37 kept, peaks at 15 GB of 25 GB -> tops up 37 -> 44, unhooks, turns compile on!
    t._peak = 15 * 1024 ** 3
    t._keep_step(768)
    assert t._keep_settle == 0
    assert t._keep_hooks is None
    assert stances == ["force_eager", "default"]
    assert t._ckpt.kept == 44
    assert keep_calls == [1, 37, 44]


def test_back_off_kept_during_eager_probe_exits_eager_and_does_not_rehook(monkeypatch):
    stances = []
    monkeypatch.setattr(T.torch.compiler, "set_stance", lambda s: stances.append(s))

    t = T.Trainer.__new__(T.Trainer)
    t._keep_auto = True
    t._compiled = 16
    t._backoffs = 0
    blocks = [nn.Linear(4, 4) for _ in range(4)]

    def fake_keep(n):
        t._ckpt.kept = n
        return n

    t._ckpt = types.SimpleNamespace(all=blocks, kept=0, keep=fake_keep)
    t._arm_eager_keep_probe()
    assert t._keep_settle == 1 and t._keep_hooks is not None and stances == ["force_eager"]

    # OOM during stage-2 verification with 4 kept: backs off eagerly (still force_eager so retry is verified eager!)
    t._keep_settle = 2
    t._ckpt.kept = 4
    t._keep_cost = {0: int(0.5 * 1024 ** 3)}
    assert t._back_off_kept() is True
    assert t._keep_settle == 2 and t._ckpt.kept == 2 and stances == ["force_eager"]

    # OOM down to 0 kept: exits eager settle immediately and restores compile stance
    t._backoffs = T.TIGHTEN_RETRIES - 1
    assert t._back_off_kept() is True
    assert t._keep_settle == 0
    assert t._keep_hooks is None
    assert t._ckpt.kept == 0
    assert stances == ["force_eager", "default"]


def test_shared_compiled_blocks_eager_probe_then_compile_gradients_match(monkeypatch):
    nodes_train = types.ModuleType("comfy_extras.nodes_train")

    def fake_patch(m, offloading=False):
        m.org_forward = m.forward
        m.forward = lambda *a, _f=m.forward, **kw: _f(*a, **kw)

    def fake_unpatch(m):
        if hasattr(m, "org_forward"):
            m.forward = m.org_forward
            delattr(m, "org_forward")

    nodes_train.patch = fake_patch
    nodes_train.unpatch = fake_unpatch
    pkg = types.ModuleType("comfy_extras")
    pkg.nodes_train = nodes_train
    monkeypatch.setitem(sys.modules, "comfy_extras", pkg)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_train", nodes_train)

    class TinyBlock(nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.w1 = nn.Linear(d, d)
            self.w2 = nn.Linear(d, d)

        def forward(self, x):
            return x + self.w2(torch.nn.functional.silu(self.w1(x)))

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([TinyBlock() for _ in range(4)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    torch.manual_seed(0)
    ref = TinyModel()
    dm = TinyModel()
    dm.load_state_dict(ref.state_dict())

    x = torch.randn(2, 8, 16)
    ref_loss = ref(x).pow(2).mean()
    ref_loss.backward()
    ref_grads = [p.grad.clone() for p in ref.parameters()]

    torch._dynamo.reset()
    with compiled_blocks(dm, True), gradient_checkpointing(dm, True, depth=1, offloading=False, skip=0, save="attention") as ck:
        # Step 1: eager probe with 1 block kept
        torch.compiler.set_stance("force_eager")
        ck.keep(1)
        dm.zero_grad(set_to_none=True)
        loss_eager = dm(x).pow(2).mean()
        loss_eager.backward()
        assert torch.allclose(loss_eager, ref_loss, atol=1e-5)

        # Jump to 2 kept (mixed compiled checkpointed + compiled un-checkpointed) and compile
        ck.keep(2)
        torch.compiler.set_stance("default")
        dm.zero_grad(set_to_none=True)
        loss_mixed = dm(x).pow(2).mean()
        loss_mixed.backward()
        assert torch.allclose(loss_mixed, ref_loss, atol=1e-5)
        for p, g in zip(dm.parameters(), ref_grads):
            assert torch.allclose(p.grad, g, rtol=1e-4, atol=1e-5)

        # Jump to all 4 un-checkpointed and run compiled
        ck.keep(4)
        dm.zero_grad(set_to_none=True)
        loss_full = dm(x).pow(2).mean()
        loss_full.backward()
        assert torch.allclose(loss_full, ref_loss, atol=1e-5)
        for p, g in zip(dm.parameters(), ref_grads):
            assert torch.allclose(p.grad, g, rtol=1e-4, atol=1e-5)
    torch._dynamo.reset()
