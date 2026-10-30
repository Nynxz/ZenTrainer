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
