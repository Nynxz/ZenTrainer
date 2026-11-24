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
    # Ceiling is set to was - 1 (3), NOT halved target (2), so if 2 succeeds with spare VRAM it can still try 3!
    t._keep_settle = 2
    t._ckpt.kept = 4
    t._keep_cost = {0: int(0.5 * 1024 ** 3)}
    assert t._back_off_kept() is True
    assert t._keep_settle == 2 and t._ckpt.kept == 2 and t._keep_ceiling == 3 and stances == ["force_eager"]

    # OOM down to 0 kept: exits eager settle immediately and restores compile stance
    t._backoffs = T.TIGHTEN_RETRIES - 1
    assert t._back_off_kept() is True
    assert t._keep_settle == 0
    assert t._keep_hooks is None
    assert t._ckpt.kept == 0
    assert stances == ["force_eager", "default"]


def test_kept_hooks_sum_across_multiple_forwards_in_one_step(monkeypatch):
    # Simulate batch_size=2 with split captions calling the same un-checkpointed block TWICE before backward
    alloc = [100 * 1024 ** 2]
    monkeypatch.setattr(T.torch.cuda, "memory_allocated", lambda: alloc[0])

    t = T.Trainer.__new__(T.Trainer)
    b = nn.Linear(4, 4)
    t._ckpt = types.SimpleNamespace(all=[b], kept=0, keep=lambda n: n)
    t._kept(1)

    orig_fwd = b.forward

    def grow_fwd(x):
        alloc[0] += 500 * 1024 ** 2  # +500 MB per forward call
        return orig_fwd(x)

    b.forward = grow_fwd
    x = torch.randn(1, 4, requires_grad=True)
    # Two forward calls in the same step -> 500 MB + 500 MB = 1000 MB total alive for backward!
    _ = b(x)
    _ = b(x)
    assert t._keep_cost[0] == 1000 * 1024 ** 2
    t._unhook_kept()

    # Step 1 landing on a 1-image bucket when batch_size=2 scales by 2x for full 2-image buckets
    t._step_key = (((1, 16, 48, 48), "None"),)
    t.batches = types.SimpleNamespace(batch_size=2)
    t.dataset = types.SimpleNamespace(buckets={(16, 48, 48): [0], (16, 48, 48): [0, 1]})
    assert t._bucket_scale() == 2.0


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
    try:
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
    finally:
        torch.compiler.set_stance("default")
        torch._dynamo.reset()


def test_deferred_eager_probe_waits_for_next_step_and_scales_already_kept_blocks(monkeypatch):
    """A probe armed by _tighten() mid-iteration must survive that iteration's _keep_step(), and
    _fit_kept() must reserve the scaled growth of already-kept blocks for larger buckets."""
    from zentrainer.harness import trainer as tr_mod
    from zentrainer.harness.trainer import Trainer

    gb = 1024 ** 3
    stances = []
    monkeypatch.setattr(torch.compiler, "set_stance", lambda s: stances.append(s))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    blocks = [nn.Linear(4, 4) for _ in range(12)]
    ck = types.SimpleNamespace(all=blocks, kept=0, allow_keep=True)
    ck.keep = lambda n: (setattr(ck, "kept", n) or n)

    t = object.__new__(Trainer)
    t._ckpt = ck
    t._compiled = 12
    t._keep_auto = True
    t._keep_warm = 0
    t._keep_cost = {}
    t._res_seen = 512
    t._peak = 10 * gb
    t._card = lambda: (20 * gb, 0)
    # 1-image current bucket vs 2-image largest bucket -> scale = 2.0
    t._step_sample_shape = (1, 16, 32, 32)
    t._step_buckets = {(16, 32, 32): [0, 1]}
    t.batches = types.SimpleNamespace(batch_size=2)

    # Mid-iteration arming from _tighten(): same-iteration _keep_step must NOT finish settling
    t._arm_eager_keep_probe(defer=True)
    assert t._keep_settle == 1 and ck.kept == 1 and stances[-1] == "force_eager"
    t._keep_step(512)
    assert t._keep_settle == 1 and t._keep_hooks is not None and t._keep_warm == 0

    # Now simulate already-kept blocks (8 kept, 0.5 GB each at 1x -> 1.0 GB each at 2x).
    # Peak at 1x = 12 GB (includes 8 * 0.5 GB = 4 GB), card = 20 GB, margin = TIGHTEN_MARGIN.
    # Extra growth of the 8 already-kept blocks at 2x = 8 * 0.5 * 1.0 * KEEP_SAFETY.
    ck.kept = 8
    t._keep_cost = {0: int(0.5 * gb)}
    t._peak = 12 * gb
    t._keep_step(512)
    # Total budget at 2x: 8 GB baseline + want * (0.5 * 2 * KEEP_SAFETY) <= 20 GB - TIGHTEN_MARGIN
    budget = 20 * gb - tr_mod.TIGHTEN_MARGIN - (12 * gb - 8 * 0.5 * gb)
    assert ck.kept * 0.5 * gb * 2.0 * tr_mod.KEEP_SAFETY <= budget + 0.5 * gb * 2.0 * tr_mod.KEEP_SAFETY
    assert ck.kept < 12  # without subtracting already-kept growth, spare=8GB-margin would jump to 12


def test_cached_model_unpatch_does_not_skip_subsequent_compile_and_save_none_shares_graph(monkeypatch):
    from zentrainer.harness.kernels.compile import compiled_frame_count

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

    class Block(nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.w1 = nn.Linear(d, d)
            self.w2 = nn.Linear(d, d)

        def forward(self, x):
            return x + self.w2(torch.nn.functional.silu(self.w1(x)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(4)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    torch.manual_seed(0)
    ref = Net()
    dm = Net()
    dm.load_state_dict(ref.state_dict())
    x = torch.randn(2, 8, 16)  # requires_grad=False (just like latent input from a frozen embedder)
    ref_loss = ref(x).pow(2).mean()
    ref_loss.backward()
    ref_grads = [p.grad.clone() for p in ref.parameters()]

    # Run 1 (uncompiled checkpointing, leaving cached model behind): unpatch must not leave stale m.__dict__["forward"]
    with gradient_checkpointing(dm, True, depth=1, offloading=False, skip=0, save="none"):
        pass
    assert all("forward" not in b.__dict__ for b in dm.blocks)
    # Even if an external unpatch left a plain bound class method on the instance, compile_blocks still compiles it
    dm.blocks[0].forward = dm.blocks[0].forward

    torch._dynamo.reset()
    try:
        before = compiled_frame_count()
        with compiled_blocks(dm, True) as n_comp, \
                gradient_checkpointing(dm, True, depth=1, offloading=False, skip=0, save="none") as ck:
            assert n_comp == 4
            # save="none" on compiled blocks shares ONE compiled checkpoint wrapper across all 4 blocks
            assert len({b.forward.__func__ for b in dm.blocks}) == 1
            dm.zero_grad(set_to_none=True)
            loss = dm(x).pow(2).mean()
            loss.backward()
            assert torch.allclose(loss, ref_loss, atol=1e-5)
            for p, g in zip(dm.parameters(), ref_grads):
                assert torch.allclose(p.grad, g, rtol=1e-4, atol=1e-5)
            # Only ONE graph compiled (not two for Block 0 requires_grad=False vs Blocks 1..3 True)
            assert compiled_frame_count() - before == 1
            assert not x.requires_grad  # caller's input tensor was never mutated
        assert all("_zt_ensure_grad" not in b.__dict__ and "forward" not in b.__dict__ for b in dm.blocks)
    finally:
        torch._dynamo.reset()


def test_first_trainable_does_not_enable_grad_on_frozen_prefix_blocks():
    class Block(nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.w = nn.Linear(d, d)

        def forward(self, x):
            return x + self.w(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(4)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    torch.manual_seed(1)
    dm = Net()
    for b in dm.blocks[:2]:
        b.requires_grad_(False)
    x = torch.randn(2, 4, 16)
    torch._dynamo.reset()
    try:
        with compiled_blocks(dm, True, first_trainable=dm.blocks[2]):
            assert not getattr(dm.blocks[0], "_zt_ensure_grad", False)
            assert getattr(dm.blocks[2], "_zt_ensure_grad", False)
            out = dm(x)
            out.pow(2).mean().backward()
            assert all(p.grad is None for b in dm.blocks[:2] for p in b.parameters())
            assert all(p.grad is not None for b in dm.blocks[2:] for p in b.parameters())
    finally:
        torch._dynamo.reset()
