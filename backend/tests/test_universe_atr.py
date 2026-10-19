"""Regression (2026-10-19): universe_atr_bars=12 (allowed 6..80) ranked NOTHING because atr_pct refused windows < 14."""
import numpy as np

from tb.risk import atr_pct
from tb.universe import Candidate, rank_candidates


def _bars(n, price=100.0, rng=1.0):
    c = np.full(n, price)
    return c + rng, c - rng, c


def test_short_ranking_windows_are_scored():
    for bars in (6, 12, 72):
        h, l, c = _bars(bars + 18)
        assert atr_pct(h, l, c, bars) is not None and atr_pct(h, l, c, bars) > 0


def test_too_little_history_is_still_none():
    h, l, c = _bars(5)
    assert atr_pct(h, l, c, 12) is None
    h, l, c = _bars(10)
    assert atr_pct(h, l, c, 72) is None          # 9 true ranges < min(72, 14)


def test_a_12_bar_window_ranks_candidates():
    h, l, c = _bars(30)
    cands = [Candidate("AAAUSDT", 1e8, 1.0, 100.0, 0.0, atr_pct(h, l, c, 12))]
    assert [x.symbol for x in rank_candidates(cands)] == ["AAAUSDT"]
