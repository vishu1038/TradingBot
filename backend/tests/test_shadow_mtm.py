"""Bots-page samples (2026-10-20): a sample taken before the marks are loaded must be skipped, not drawn without the
open positions (that made a fake drop at every restart)."""
import pytest

from test_engine_core import mk_engine
from tb.shadow import ShadowPos


def _pos(sym="AAAUSDT"):
    return ShadowPos(model="m", symbol=sym, side="long", entry=100.0, qty=2.0, sl=99.0, tp=103.0, opened_at=0,
                     prob=0.5, entry_fee=0.1)


def test_no_sample_while_an_open_position_has_no_mark(store, tmp_path):
    eng, _ = mk_engine(store, tmp_path)
    eng.shadow.pos["m|ZZZUSDT"] = _pos("ZZZUSDT")
    eng.marks.pop("ZZZUSDT", None)
    assert eng.shadow_mtm() == {}


def test_open_position_is_marked_net_of_its_entry_fee(store, tmp_path):
    eng, _ = mk_engine(store, tmp_path)
    eng.shadow.pos["m|AAAUSDT"] = _pos()
    eng.marks["AAAUSDT"] = 101.0
    assert eng.shadow_mtm()["m"] == pytest.approx(2.0 * 1.0 - 0.1)
