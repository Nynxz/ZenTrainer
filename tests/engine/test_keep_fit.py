"""memory.grad_checkpoint_skip "auto" across a resolution stage change (harness/trainer.py): the kept
blocks' fit counts only steps at the highest resolution so far, so a blend's lower-resolution steps
can't grow it past what the higher one holds; and an out-of-VRAM step that the margin's worth didn't
fix halves the kept blocks rather than dropping one at a time."""

import types

from zentrainer.harness import trainer as T


def _trainer(kept=0):
    t = T.Trainer.__new__(T.Trainer)
    t._keep_auto, t._keep_warm, t._res_seen, t._keep_cost = True, 0, 0, {0: 1024 ** 3 // 2}
    t._ckpt = types.SimpleNamespace(kept=kept)
    t.fits = []
    t._fit_kept = lambda: t.fits.append(t._keep_warm)

    def kept_n(n):
        t._ckpt.kept = n
        return n

    t._kept = kept_n
    return t


def test_a_blends_lower_resolution_steps_dont_count():
    t = _trainer()
    t._res_seen = 1024  # a 1024 step has come: the blend has begun
    for res in [768, 1024, 768, 768, 1024, 768, 1024]:
        t._keep_step(res)
    assert t._keep_warm == 3 and t.fits == [1, 3]  # KEEP_AT's 1 and 3: the three 1024 steps only
    t2 = _trainer()
    t2._res_seen = 768
    for _ in range(3):
        t2._keep_step(768)  # one resolution: every step counts, as before
    assert t2._keep_warm == 3


def test_an_out_of_vram_step_halves_the_kept_blocks_once_the_margin_wasnt_enough():
    t = _trainer(kept=10)
    t._backoffs = 0
    assert t._back_off_kept() and t._ckpt.kept == 8  # first: the margin's worth (1 GB / 0.575 GB a block, 2)
    t._backoffs = 1
    assert t._back_off_kept() and t._ckpt.kept == 4  # again: half
    assert t._keep_ceiling == 4  # and never more again
    t._backoffs = T.TIGHTEN_RETRIES - 1
    assert t._back_off_kept() and t._ckpt.kept == 0  # the last retry tries none kept


def test_a_new_resolution_measures_its_peak_afresh():
    # weights all on the card (no auto VRAM fit): the kept blocks reset, and the peak with them, or
    # the lower stage's peak leaves the higher one no room (Krea 2: none kept at 1024)
    t = _trainer(kept=13)
    t._peak, t._keep_warm, t._keep_cost = 24 * 1024 ** 3, 20, {0: 1}
    t._auto_vram = lambda: False
    t._new_resolution(1024)
    assert (t._ckpt.kept, t._peak, t._keep_warm, t._keep_cost) == (0, 0, 0, {})


def test_free_vram_unlocks_save_attention_and_keep_auto_and_backs_off_in_priority_order(monkeypatch):
    import sys

    t = _trainer(kept=0)
    t._want_keep_auto, t._keep_auto = True, False
    t.spec = types.SimpleNamespace(memory=types.SimpleNamespace(checkpoint_save="attention"))
    t._ckpt.all = [object(), object()]
    t._ckpt.allow_keep = False
    t._ckpt.saving_attention = False

    def set_save(on):
        if on == t._ckpt.saving_attention:
            return False
        t._ckpt.saving_attention = on
        return True

    t._ckpt.set_save_attention = set_save
    model = types.SimpleNamespace(model_lowvram=True)
    patcher = types.SimpleNamespace(model=model, loaded_size=lambda: 8 << 30, model_size=lambda: 10 << 30)
    t.models = types.SimpleNamespace(patcher=patcher)
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)

    # still partially offloaded: both stay off
    t._maybe_use_free_vram(spare=4 << 30)
    assert not t._keep_auto and not t._ckpt.allow_keep and not t._ckpt.saving_attention

    # 100% resident, but less than 1 GB spare: keep_auto unlocks, saving_attention waits
    model.model_lowvram = False
    patcher.loaded_size = lambda: 10 << 30
    t._maybe_use_free_vram(spare=256 << 20)
    assert t._keep_auto and t._ckpt.allow_keep and not t._ckpt.saving_attention

    # >= 1 GB spare: saving_attention turns on too
    t._maybe_use_free_vram(spare=2 << 30)
    assert t._ckpt.saving_attention

    # OOM priority 1: drop kept blocks first (saving_attention and weights stay untouched)
    t._ckpt.kept = 6
    t._keep_cost = {0: 1024 ** 3 // 2}
    t._backoffs = 0
    t._auto_vram = lambda: True
    fit_calls = []
    t._fit_weights = lambda: fit_calls.append(1)
    t.optimizer = types.SimpleNamespace(zero_grad=lambda set_to_none=True: None)
    mm = types.ModuleType("comfy.model_management")
    mm.soft_empty_cache = lambda: None
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(T.torch.cuda, "reset_peak_memory_stats", lambda: None)

    assert t._back_off(RuntimeError("oom"))
    assert t._ckpt.kept == 4 and t._ckpt.saving_attention and not fit_calls

    # OOM priority 2: once kept == 0, turn off saving_attention before offloading any weights
    t._ckpt.kept = 0
    assert t._back_off(RuntimeError("oom"))
    assert not t._ckpt.saving_attention and getattr(t, "_save_attn_off", False) and not fit_calls

    # OOM priority 3: once saving_attention is off, offload model weights
    t._recompile_if_moved = lambda why="": None
    assert t._back_off(RuntimeError("oom"))
    assert len(fit_calls) == 1 and not t._keep_auto and not t._ckpt.allow_keep


def test_gradient_checkpointing_toggles_save_attention_and_respects_sharp_qk(monkeypatch):
    import sys

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from zentrainer.harness.memory import gradient_checkpointing

    # Minimal stub for comfy_extras.nodes_train patch/unpatch
    nodes_train = types.ModuleType("comfy_extras.nodes_train")

    def fake_patch(m, offloading=False):
        org = m.forward
        m.org_forward = org
        m.forward = lambda *a, **kw: torch.utils.checkpoint.checkpoint(org, *a, use_reentrant=False, **kw)

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
        def __init__(self, sharp=False):
            super().__init__()
            self.q_norm = nn.LayerNorm(16)
            self.k_norm = nn.LayerNorm(16)
            if sharp:
                with torch.no_grad():
                    self.q_norm.weight.fill_(30.0)
                    self.k_norm.weight.fill_(30.0)
            self.proj = nn.Linear(16, 16)

        def forward(self, x):
            q = self.q_norm(x).unsqueeze(1)
            k = self.k_norm(x).unsqueeze(1)
            v = x.unsqueeze(1)
            return self.proj(F.scaled_dot_product_attention(q, k, v).squeeze(1))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block(sharp=True), Block(sharp=False), Block(sharp=False)])

        def forward(self, x):
            for b in self.blocks:
                x = x + b(x)
            return x

    model = Model()
    x0 = torch.randn(2, 8, 16, requires_grad=True)
    ref = model(x0).sum()
    ref.backward()
    ref_grad = x0.grad.clone()
    model.zero_grad()

    with gradient_checkpointing(model, enabled=True, depth=1, offloading=True, skip=0, save="attention") as ck:
        assert not ck.saving_attention and ck.blocks == 3
        # Dynamically enable selective attention saving (as _tighten does once weights fit)
        assert ck.set_save_attention(True) and ck.saving_attention
        assert not ck.set_save_attention(True)  # idempotent
        x1 = x0.detach().clone().requires_grad_(True)
        model(x1).sum().backward()
        assert torch.allclose(x1.grad, ref_grad, atol=1e-5, rtol=1e-4)

        # Dynamically disable selective attention saving (as _back_off does on OOM)
        assert ck.set_save_attention(False) and not ck.saving_attention
        x2 = x0.detach().clone().requires_grad_(True)
        model(x2).sum().backward()
        assert torch.allclose(x2.grad, ref_grad, atol=1e-5, rtol=1e-4)

    # All org_forward hooks cleanly restored on exit
    assert all(not hasattr(b, "org_forward") for b in model.blocks)


def test_patch_offload_block_works_with_torch_compile():
    import torch
    import torch.nn as nn

    from zentrainer.harness.memory import _patch_offload_block

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(16, 16)
            self.l2 = nn.Linear(16, 16)

        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))

    b = Block()
    x = torch.randn(2, 4, 16, requires_grad=True)
    ref = b(x).sum()
    ref.backward()
    ref_x = x.grad.clone()
    ref_w = b.l1.weight.grad.clone()
    b.zero_grad()

    b.forward = torch.compile(b.forward, backend="aot_eager", dynamic=True)
    _patch_offload_block(b)
    x2 = x.detach().clone().requires_grad_(True)
    b(x2).sum().backward()
    assert torch.allclose(x2.grad, ref_x, atol=1e-5, rtol=1e-5)
    assert torch.allclose(b.l1.weight.grad, ref_w, atol=1e-5, rtol=1e-5)
