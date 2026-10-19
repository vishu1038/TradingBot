"""ML-1 artifact rules the engine must enforce: labels-exact ATR, time stop (exit_cfg.horizon_bars), max_open,
and the same rules in the shadow simulation."""
import pytest

from conftest import make_klines
from test_engine_core import BAR, mk_engine, run
from test_v3_engine_core import doubles, hour_now, write_v3  # noqa: F401  (doubles = fixture)
from tb import features_mtf as fm
from tb.config import Settings
from tb.entry import EntryCfg, EntryGate
from tb.exchange.base import SymbolFilters
from tb.exits import ExitSpec, exit_atr, resolve_exit
from tb.labels import exit_atr as label_exit_atr
from tb.shadow import ShadowBook

PCT = {"exit_mode": "pct", "tp_pct": 3.0, "sl_pct": 6.0, "horizon_bars": 12}


def test_atr_for_a_higher_timeframe_is_exactly_the_labels_atr():
    now_ms = int(hour_now() * 1000)
    kl = make_klines(8000, end_ms=now_ms)
    closed = kl[kl["open_time"] + BAR <= now_ms].reset_index(drop=True)
    for tf in ("1h", "4h", "1d"):
        want = float(fm.htf_atr_5m(closed, tf, 14)[-1])
        assert exit_atr(ExitSpec("atr", tf, tp_atr=2.0, sl_atr=1.0), closed, now_ms) == pytest.approx(want, rel=1e-12)
        assert want == pytest.approx(float(label_exit_atr(closed, {"atr_tf": tf})[-1]))
    assert resolve_exit({"exit_mode": "atr", "atr_tf": "2h"}).mode == "settings"        # not a labels timeframe


def test_horizon_is_parsed_for_every_spec():
    assert resolve_exit(PCT).horizon_bars == 12
    assert resolve_exit({"exit_mode": "atr", "tp_atr": 3, "sl_atr": 2, "horizon_bars": 288}).horizon_bars == 288
    assert resolve_exit({"exit_mode": "atr", "tp_atr": 3, "sl_atr": 2}).horizon_bars is None


def test_live_time_stop_closes_at_market_after_the_horizon_and_survives_a_restart(store, tmp_path, doubles):
    write_v3(tmp_path / "models", "ts", "champion", kind="mtf", history_bars_5m=100, exit_cfg=PCT)
    eng, _ = mk_engine(store, tmp_path)
    run(eng.bars_cycle())
    pos = store.get_positions()
    assert len(pos) == 2
    p = pos[0]
    t0 = p["opened_at"]
    assert eng.time_stops[str(p["id"])] == t0 + 12 * 300
    marks = {x["symbol"]: x["entry_price"] for x in pos}
    eng.clock = lambda: t0 + 12 * 300 - 5
    async def sync():
        async with eng.tlock:
            await eng.sync_positions(marks)
    run(sync())
    assert len(store.get_positions()) == 2                                   # before the horizon: untouched
    eng.persist_state(force=True)
    assert store.kv_get("time_stops")                                        # persisted for a restart
    eng.time_stops = {}
    eng.time_stops = {k: float(v) for k, v in store.kv_get("time_stops").items()}
    eng.clock = lambda: t0 + 12 * 300 + 1
    run(sync())
    assert store.get_positions() == []
    tr = store.get_trades(5)
    assert {t["exit_reason"] for t in tr} == {"timeout"} and eng.time_stops == {}
    assert eng.ex.positions() == [] and eng.ex.open_conditional_orders() == []   # stops cleaned up with the close


def test_shadow_time_stop_closes_at_the_bar_close_with_slippage_and_fees():
    cfg = Settings().merge({"slippage_pct": 0.001, "stop_slippage_pct": 0.0}, allow_internal=True)
    f = SymbolFilters(0.01, 0.001, 0.001, 5.0)
    spec = resolve_exit(PCT)
    b = ShadowBook(cfg)
    t0 = 1_700_000_000.0
    assert b.open("m", "A", "long", 0.9, 100.0, t0, None, f, spec=spec)
    p = b.pos["m|A"]
    assert p.deadline == int(t0) + 12 * 300
    # bars inside the horizon: nothing happens
    assert b.on_bar("A", int((t0 + 3000) * 1000), 99.0, 101.0, t0 + 3300, close=100.5) == []
    # the bar that ENDS at the deadline: closed at its close (adverse slippage), reason timeout, fees both sides
    open_ms = int((p.deadline - 300) * 1000)
    out = b.on_bar("A", open_ms, 99.0, 101.0, t0 + 4000, close=101.0)
    assert len(out) == 1 and out[0]["exit_reason"] == "timeout"
    assert out[0]["exit_price"] == pytest.approx(101.0 * (1 - 0.001))
    assert out[0]["pnl"] == pytest.approx((out[0]["exit_price"] - p.entry) * p.qty - out[0]["fees"])
    # tick path: a live price at/after the deadline also closes it
    b.open("m", "B", "short", 0.9, 100.0, t0, None, f, spec=spec)
    assert b.on_price("B", 100.0, t0 + 3599) == []
    r = b.on_price("B", 100.0, t0 + 3600)
    assert len(r) == 1 and r[0]["exit_reason"] == "timeout"


def test_max_open_gate_unit():
    g = EntryGate()
    c = EntryCfg.resolve(Settings(), {"max_open": 5})
    assert c.max_open == 5 and EntryCfg.resolve(Settings(), {}).max_open is None
    assert g.check("A", 1.0, c, 4)[0]
    ok, why = g.check("A", 1.0, c, 5)
    assert not ok and "max_open" in why


def test_champion_never_holds_more_than_max_open(store, tmp_path, doubles):
    write_v3(tmp_path / "models", "mo", "champion", kind="mtf", history_bars_5m=100, entry_cfg={"max_open": 1})
    eng, _ = mk_engine(store, tmp_path)
    run(eng.bars_cycle())
    assert len(store.get_positions()) == 1
    eng.last_closed.clear()
    run(eng.bars_cycle())                                                    # still signalling, still capped
    assert len(store.get_positions()) == 1


def test_shadow_challenger_obeys_max_open_and_time_stop(store, tmp_path, doubles):
    d = tmp_path / "models"
    write_v3(d, "champ", "champion", kind="mtf", history_bars_5m=100)
    write_v3(d, "rules", "challenger", kind="mtf", history_bars_5m=100, entry_cfg={"max_open": 1}, exit_cfg=PCT)
    eng, _ = mk_engine(store, tmp_path)
    run(eng.bars_cycle())
    mine = [p for p in eng.shadow.pos.values() if p.model == "rules"]
    assert len(mine) == 1 and mine[0].deadline == mine[0].opened_at + 12 * 300
    assert sum(1 for p in eng.shadow.pos.values() if p.model == "champ") == 2        # the champion's shadow is uncapped
