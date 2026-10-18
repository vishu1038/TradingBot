"""Engine behaviour against a scripted market-data source and the paper exchange."""
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import FakeData, make_klines
from tb import engine as E
from tb import guard as G
from tb import scorer as S
from tb.alerts import AlertDispatcher
from tb.config import Settings
from tb.exchange.base import ExchangeError
from tb.exchange.paper import PaperExchange
from tb.safety import DEGRADED, NORMAL

BAR = 300_000
FAST = {"slippage_pct": 0.0, "stop_slippage_pct": 0.0}


def run(coro):
    return asyncio.run(coro)


class Sent:
    """Alert sink."""
    name = "mem"

    def __init__(self):
        self.msgs = []

    def send(self, text):
        self.msgs.append(text)


class FixedScorer:
    name = "fixed"
    version = "1"

    def __init__(self, pl=0.9, ps=0.0):
        self.pl, self.ps = pl, ps

    def score(self, symbol, feats):
        return S.Score(self.pl, self.ps, [S.Reason("ret_1", "5m return", 0.1, 0.05)])


def mk_engine(store, tmp_path, *, mode="paper_local", settings=None, now=None, symbols=("AAAUSDT", "BBBUSDT", "BTCUSDT")):
    boundary = int(time.time() // 300 * 300)
    now = now if now is not None else boundary + 10.0
    data = FakeData(symbols)
    for i, s in enumerate(symbols):
        data.kl[s] = make_klines(base=100.0, seed=i + 1, end_ms=int(now * 1000))
        data.marks[s] = float(data.kl[s]["close"].iloc[-1])
    data.server_time_ms = lambda: int(now * 1000)
    if settings:
        store.patch_settings(settings, allow_internal=True)
    cfg = store.get_settings().merge(FAST, allow_internal=True)
    store.save_settings(cfg)
    ex = PaperExchange(data, cfg)
    sink = Sent()
    eng = E.Engine(store, data, ex, mode, AlertDispatcher([sink], min_interval=0), models_path=tmp_path / "models",
                   clock=lambda: now)
    eng.sink = sink
    eng.startup()
    eng.universe = list(symbols[:2])
    eng.running = True
    return eng, data


def closed_of(eng, sym):
    k = eng.kl[sym]
    return k[k["open_time"] + BAR <= eng.now_ms()]


def load_bars(eng, syms=None):
    async def go():
        for s in syms or eng.watchlist():
            await eng.update_symbol(s)
    run(go())


def open_long(eng, sym="AAAUSDT", prob=0.9):
    async def go():
        await eng.update_symbol(sym)
        await eng.try_open(sym, "long", prob, [{"feature": "f", "label": "l", "value": 1.0, "contribution": 0.1}],
                           closed_of(eng, sym))
    run(go())


# ------------------------------------------------------------------ opening
def test_open_places_stop_and_tp_and_records_state(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng)
    pos = store.get_positions()
    assert len(pos) == 1
    p = pos[0]
    assert p["symbol"] == "AAAUSDT" and p["sl_order_id"] and p["tp_order_id"]
    px = p["entry_price"]
    assert p["qty"] * px == pytest.approx(160.0, rel=0.01)
    assert (px - p["sl_price"]) * p["qty"] == pytest.approx(10.0, abs=0.1)      # -10 USD at the stop
    assert (p["tp_price"] - px) * p["qty"] == pytest.approx(5.0, abs=0.1)       # +5 USD at the target
    orders = eng.ex.open_conditional_orders("AAAUSDT")
    assert {o.kind for o in orders} == {"SL", "TP"} and all(o.reduce_only for o in orders)
    types = [e["type"] for e in store.events_after(0)]
    assert "position_opened" in types
    eng.alerts.flush(2)
    assert any("OPEN LONG AAAUSDT" in m for m in eng.sink.msgs)
    assert store.kv_get("paper_state")["pos"]["AAAUSDT"]


def test_sl_placement_failure_closes_position_immediately(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)

    def boom(*a):
        raise ExchangeError("stop rejected", code=-2021)

    eng.ex.place_stop = boom
    open_long(eng)
    assert store.get_positions() == []
    assert eng.ex.positions() == []                      # never left unprotected
    assert eng.ex.balance().wallet < 100.0               # only fees/slippage lost
    eng.alerts.flush(2)
    assert any("sl_placement" in m for m in eng.sink.msgs)


def test_stop_invariant_closes_position_without_live_stop(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng)
    eng.ex.conds.clear()                                 # simulate a lost stop (and TP)
    run(eng.prices_cycle())
    assert store.get_positions() == []
    assert eng.ex.positions() == []
    t = store.get_trades(1)[0]
    assert t["exit_reason"] == "manual"


def test_tp_hit_recorded_with_fees(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng)
    p = store.get_positions()[0]
    data.marks["AAAUSDT"] = p["tp_price"] * 1.001
    run(eng.prices_cycle())
    assert store.get_positions() == []
    t = store.get_trades(1)[0]
    assert t["exit_reason"] == "tp" and t["fees"] > 0
    assert t["pnl"] == pytest.approx(5.0 - t["fees"] - t["funding"], abs=0.1)     # pnl is NET, fees/funding are costs
    assert t["pnl"] == pytest.approx(eng.ex.balance().wallet - 100.0)             # equals the wallet change
    assert eng.ex.open_conditional_orders() == []        # leftover orders cancelled
    assert "position_closed" in [e["type"] for e in store.events_after(0)]


def test_sl_hit_and_trail_reason(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng)
    p = store.get_positions()[0]
    data.marks["AAAUSDT"] = p["sl_price"] * 0.999
    run(eng.prices_cycle())
    assert store.get_trades(1)[0]["exit_reason"] == "sl"
    # same but with trailing active on the row -> reported as "trail"
    eng2, data2 = mk_engine(store, tmp_path)
    open_long(eng2)
    p = store.get_positions()[0]
    store.update_position(p["id"], trail_active=True)
    data2.marks["AAAUSDT"] = p["sl_price"] * 0.999
    run(eng2.prices_cycle())
    assert store.get_trades(1)[0]["exit_reason"] == "trail"


def test_trailing_moves_stop_up_never_down(store, tmp_path):
    eng, data = mk_engine(store, tmp_path, settings={"trail": {"enabled": True, "activate_r": 0.5, "distance_atr": 0.5}})
    eng.cfg = store.get_settings()
    open_long(eng)
    p = store.get_positions()[0]
    sl0 = p["sl_price"]
    entry = p["entry_price"]
    risk = entry - sl0
    atr = p["atr"]
    data.marks["AAAUSDT"] = entry + 0.8 * risk              # well past +0.5R but below the TP (+0.5 * risk)... keep TP away
    store.update_position(p["id"], tp_price=entry + 10 * risk)
    eng.ex.conds = {k: c for k, c in eng.ex.conds.items() if c.kind == "SL"}
    store.update_position(p["id"], tp_order_id=None)
    run(eng.prices_cycle())
    p2 = store.get_positions()[0]
    assert p2["trail_active"] and p2["sl_price"] > sl0
    assert [o.trigger_price for o in eng.ex.open_conditional_orders("AAAUSDT")] == [p2["sl_price"]]   # old stop replaced
    assert p2["sl_price"] < data.marks["AAAUSDT"]
    new_sl = p2["sl_price"]
    data.marks["AAAUSDT"] = data.marks["AAAUSDT"] - 0.1 * atr   # small pull-back, still above the trailed stop
    run(eng.prices_cycle())
    p3 = store.get_positions()[0]
    assert p3["sl_price"] == new_sl                              # never loosened
    data.marks["AAAUSDT"] = new_sl * 0.999                        # through the trailed stop -> exit as "trail"
    run(eng.prices_cycle())
    assert store.get_positions() == [] and store.get_trades(1)[0]["exit_reason"] == "trail"
    assert store.get_trades(1)[0]["pnl"] > 0


# ------------------------------------------------------------------ gating
def test_entry_blocked_by_running_kill_safety_and_exclusion(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    assert eng.entry_allowed("AAAUSDT")
    eng.running = False
    assert not eng.entry_allowed("AAAUSDT")
    eng.running = True
    eng.kill = {"tripped": True}
    assert not eng.entry_allowed("AAAUSDT")
    eng.kill = {"tripped": False}
    eng.safety_view = {"state": DEGRADED}
    open_long(eng)
    assert store.get_positions() == []                   # try_open obeys safety state
    eng.safety_view = {"state": NORMAL}
    for _ in range(3):
        eng.faults.observe_symbol("AAAUSDT", ["big_move"])
    assert not eng.entry_allowed("AAAUSDT") and eng.entry_allowed("BBBUSDT")


def test_guard_state_in_store_blocks_entries_after_heartbeat_cycle(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    store.kv_set(G.KEY_STATE, {"state": "FLATTEN", "reasons": ["x"], "t": time.time()})
    run(eng.heartbeat_cycle())
    assert eng.safety_view["state"] == "FLATTEN" and not eng.entry_allowed("AAAUSDT")
    store.kv_set(G.KEY_STATE, {"state": "NORMAL", "reasons": []})
    run(eng.heartbeat_cycle())
    assert eng.entry_allowed("AAAUSDT")
    # demo/live also require a live guard process
    eng2, _ = mk_engine(store, tmp_path, mode="binance_demo")
    run(eng2.heartbeat_cycle())
    assert eng2.safety_view["state"] == DEGRADED and "guard" in " ".join(eng2.safety_view["reasons"])
    store.kv_set(G.KEY_BEAT, {"t": time.time(), "pid": 1})
    run(eng2.heartbeat_cycle())
    assert eng2.safety_view["state"] == NORMAL


def test_correlated_positions_block_new_entry(store, tmp_path):
    syms = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "BTCUSDT")
    eng, data = mk_engine(store, tmp_path, symbols=syms, settings={"corr_cap": {"max_corr": 0.8, "max_correlated": 2}})
    eng.universe = list(syms[:4])
    base = data.kl["AAAUSDT"]
    for s in ("BBBUSDT", "CCCUSDT", "DDDUSDT"):           # near-identical return paths -> correlation ~1
        k = base.copy()
        noise = np.random.default_rng(len(s)).normal(0, 1e-5, len(k))
        for c in ("open", "high", "low", "close"):
            k[c] = base[c] * (1 + noise)
        data.kl[s] = k
        data.marks[s] = float(k["close"].iloc[-1])
    open_long(eng, "AAAUSDT")
    open_long(eng, "BBBUSDT")
    open_long(eng, "CCCUSDT")
    assert {p["symbol"] for p in store.get_positions()} == {"AAAUSDT", "BBBUSDT"}      # 3rd blocked: 2 already correlated
    open_long(eng, "DDDUSDT")
    assert len(store.get_positions()) == 2


def test_fifth_slot_unreachable_with_100_equity(store, tmp_path):
    syms = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "BTCUSDT")
    eng, data = mk_engine(store, tmp_path, symbols=syms)
    eng.universe = list(syms[:5])
    for s in syms[:5]:
        open_long(eng, s)
    assert len(store.get_positions()) == 4                       # the 5th slice is short by the entry fees
    run(eng.heartbeat_cycle())
    assert store.kv_get("status")["max_positions"] == 4


def test_max_positions_follows_equity(store, tmp_path):
    eng, data = mk_engine(store, tmp_path, settings={"paper_start_equity": 45.0})
    eng.startup()
    assert E.effective_slots(eng.cfg, 0, eng.available) == 2          # 45 / 20.08


# ------------------------------------------------------------------ derating / staged rollout / drift
def test_drawdown_derating_halves_size(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng, "AAAUSDT")
    full = store.get_positions()[0]["qty"]
    eng2, _ = mk_engine(store, tmp_path)
    store.execute("DELETE FROM positions")
    eng2.ex.pos.clear()
    eng2.peak_equity = eng2.equity / 0.6                  # 40% drawdown vs the 50% paper kill limit -> x0.5
    mult, parts = eng2.risk_multiplier()
    assert mult == pytest.approx(0.5) and parts["dd_pct"] == pytest.approx(40.0)
    open_long(eng2, "AAAUSDT")
    half = store.get_positions()[0]["qty"]
    assert half == pytest.approx(full / 2, rel=0.02)


def test_derated_trade_below_min_notional_is_skipped(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    eng.ex.exchange_info = lambda: {s: __import__("tb.exchange.base", fromlist=["SymbolInfo"]).SymbolInfo(
        s, filters=__import__("tb.exchange.base", fromlist=["SymbolFilters"]).SymbolFilters(0.01, 0.001, 0.001, 100.0))
        for s in data.marks}
    eng.peak_equity = eng.equity / 0.6                    # x0.5 -> 80 notional < 100 minimum
    open_long(eng)
    assert store.get_positions() == []


def test_live_stage_half_size_for_first_20_trades(store, tmp_path):
    eng, _ = mk_engine(store, tmp_path, mode="live")
    assert eng.risk_multiplier()[0] == pytest.approx(0.5)
    store.kv_set("live_trades_opened", 20)
    assert eng.risk_multiplier()[0] == pytest.approx(1.0)
    eng_p, _ = mk_engine(store, tmp_path)
    assert eng_p.risk_multiplier()[0] == 1.0              # never applies outside live


def _trade(store, prob, win, i=0):
    store.insert_trade({"symbol": "A", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": i,
                        "closed_at": i + 1, "pnl": 1.0 if win else -1.0, "fees": 0.0, "exit_reason": "tp" if win else "sl",
                        "prob": prob})


def test_drift_halves_then_blocks_entries(store, tmp_path):
    eng, _ = mk_engine(store, tmp_path)
    for i in range(30):
        _trade(store, 0.6, i < 12, i)                     # realised 40% vs predicted 60%: z ~ -2.2
    eng._update_drift()
    assert eng.drift.level == 1 and eng.risk_multiplier()[0] == pytest.approx(0.5)
    for i in range(30, 60):
        _trade(store, 0.6, i < 39, i)                     # last 30: 9/30 = 30% -> z ~ -3.3
    eng._update_drift()
    assert eng.drift.level == 2
    run(eng.heartbeat_cycle())
    assert eng.safety_view["state"] == DEGRADED and not eng.entry_allowed("AAAUSDT")
    assert store.get_trades(1)  # sanity


def test_slippage_drift_halves_size(store, tmp_path):
    eng, _ = mk_engine(store, tmp_path)
    eng.cfg = eng.cfg.merge({"slippage_pct": 0.0002}, allow_internal=True)
    eng.slips.extend([0.1] * 10)                          # 0.1% measured vs 0.02% assumed
    eng.slip_alarm = E.slippage_drift(list(eng.slips), eng.cfg.slippage_pct * 100, eng.cfg.slippage_drift_mult)
    assert eng.risk_multiplier()[0] == pytest.approx(0.5)


# ------------------------------------------------------------------ kill switch
def test_kill_switch_closes_all_and_blocks_until_reset(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng, "AAAUSDT")
    open_long(eng, "BBBUSDT")
    assert len(store.get_positions()) == 2
    # crash the price: equity falls through the 50% paper kill limit
    for s in ("AAAUSDT", "BBBUSDT"):
        data.marks[s] = data.marks[s] * 0.5
    eng.peak_equity = 100.0
    eng.ex.wallet = 52.0
    run(eng.prices_cycle())
    assert eng.kill["tripped"] and store.get_positions() == []
    assert {t["exit_reason"] for t in store.get_trades(5)} <= {"kill", "liquidation", "sl"}
    assert any(e["type"] == "kill" for e in store.events_after(0))
    eng.alerts.flush(2)
    assert any("KILL SWITCH" in m for m in eng.sink.msgs)
    assert not eng.entry_allowed("AAAUSDT")
    st = store.kv_get("status")
    assert st["kill"]["tripped"] is True
    # start is refused while tripped; reset_kill re-baselines the peak and signals the guard
    err = run(eng.handle_command("start", {}))
    assert err and "reset_kill" in err
    assert run(eng.handle_command("reset_kill", {})) is None
    assert not eng.kill["tripped"] and eng.peak_equity == pytest.approx(eng.equity)
    assert store.kv_get(G.KEY_RESET) is not None
    assert eng.entry_allowed("AAAUSDT")


def test_kill_switch_triggers_exactly_at_threshold(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    eng.peak_equity = 100.0
    eng.ex.wallet = 50.5
    run(eng.prices_cycle())
    assert not eng.kill["tripped"]                       # 49.5% < 50%
    eng.ex.wallet = 50.0
    run(eng.prices_cycle())
    assert eng.kill["tripped"]


# ------------------------------------------------------------------ commands
def test_command_flow(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng, "AAAUSDT")
    pid = store.get_positions()[0]["id"]
    store.add_command("stop")
    store.add_command("close_position", {"position_id": 9999})
    store.add_command("close_position", {"position_id": pid})
    store.add_command("bogus")
    run(eng.commands_cycle())
    assert eng.running is False and store.get_positions() == []
    assert store.pending_commands() == []
    res = [store.command_result(i) for i in (1, 2, 3, 4)]
    assert res[0]["ok"] and "no such position" in res[1]["error"] and res[2]["ok"] and "unknown action" in res[3]["error"]
    store.add_command("start")
    run(eng.commands_cycle())
    assert eng.running is True and store.kv_get("running") is True


def test_close_all_command(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng, "AAAUSDT")
    open_long(eng, "BBBUSDT")
    store.add_command("close_all")
    run(eng.commands_cycle())
    assert store.get_positions() == [] and len(store.get_trades(5)) == 2


# ------------------------------------------------------------------ bars / signals / plausibility
def test_new_bar_scores_records_signal_and_opens(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    eng.scorer = FixedScorer(0.9)
    eng.model_info = S.ModelInfo("fixed", "1", None, 0.7, 0.8)
    run(eng.bars_cycle())
    sig = store.get_signals("AAAUSDT")
    assert sig and sig[-1]["action"] == "long" and sig[-1]["prob_long"] == 0.9 and sig[-1]["reasons"][0]["feature"] == "ret_1"
    assert {p["symbol"] for p in store.get_positions()} == {"AAAUSDT", "BBBUSDT"}
    # candles were written for the UI; each bar signal only once
    assert len(store.get_candles("AAAUSDT", "5m", 1000)) >= 600
    n = len(store.get_signals(None, 100))
    run(eng.bars_cycle())
    assert len(store.get_signals(None, 100)) == n


def test_threshold_and_short_rules(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    eng.scorer = FixedScorer(0.69, 0.79)
    eng.model_info = S.ModelInfo("fixed", "1", None, 0.7, 0.8)
    run(eng.bars_cycle())
    assert store.get_positions() == [] and {s["action"] for s in store.get_signals(None, 50)} == {"none"}
    eng2, _ = mk_engine(store, tmp_path, settings={"short_enabled": True})
    eng2.scorer = FixedScorer(0.1, 0.85)
    eng2.model_info = eng.model_info
    run(eng2.bars_cycle())
    assert {p["side"] for p in store.get_positions()} == {"short"}


def test_nullscorer_never_trades_but_records_signals(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    run(eng.bars_cycle())
    assert store.get_positions() == [] and store.get_signals(None, 10)
    assert all(s["prob_long"] == 0 and s["action"] == "none" for s in store.get_signals(None, 10))


def test_implausible_bar_rejected_then_symbol_excluded(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    eng.scorer = FixedScorer(0.9)
    eng.model_info = S.ModelInfo("fixed", "1", None, 0.7, 0.8)
    k = data.kl["AAAUSDT"].copy()
    k.loc[k.index[-2], "close"] = k["close"].iloc[-3] * 1.5        # +50% in one closed bar
    k.loc[k.index[-2], "high"] = k["close"].iloc[-2] * 1.001
    data.kl["AAAUSDT"] = k
    run(eng.bars_cycle())
    assert store.get_signals("AAAUSDT") == [] and "AAAUSDT" not in {p["symbol"] for p in store.get_positions()}
    assert "BBBUSDT" in {p["symbol"] for p in store.get_positions()}     # others unaffected
    assert not eng.faults.excluded("AAAUSDT")                      # warn first ...
    run(eng.bars_cycle())
    run(eng.bars_cycle())
    assert eng.faults.excluded("AAAUSDT")                          # ... excluded once matured (3 consecutive)
    run(eng.heartbeat_cycle())
    assert store.kv_get("safety_engine")["faults"][0]["symbol"] == "AAAUSDT"
    data.kl["AAAUSDT"] = make_klines(base=100.0, seed=1, end_ms=int(eng.clock() * 1000))     # data heals
    for _ in range(5):
        run(eng.bars_cycle())
    assert not eng.faults.excluded("AAAUSDT")                      # cleared after 5 good cycles (hysteresis)


def test_exchange_outage_survives_and_escalates_to_degraded(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng, "AAAUSDT")
    data.fail = True
    for _ in range(3):
        run(eng.bars_cycle())                                      # must not raise
        with pytest.raises(Exception):
            run(eng.prices_cycle())
    run(eng.heartbeat_cycle())
    assert eng.safety_view["state"] == DEGRADED and "exchange-wide" in " ".join(eng.safety_view["reasons"])
    assert len(store.get_positions()) == 1                         # outage never force-closes (stop is on the exchange)
    data.fail = False
    for _ in range(5):
        run(eng.prices_cycle())
        run(eng.bars_cycle())
    run(eng.heartbeat_cycle())
    assert eng.safety_view["state"] == NORMAL


def test_one_bad_symbol_does_not_stop_the_loop(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    eng.scorer = FixedScorer(0.9)
    eng.model_info = S.ModelInfo("fixed", "1", None, 0.7, 0.8)
    orig = data.klines

    def flaky(symbol, *a, **k):
        if symbol == "AAAUSDT":
            raise RuntimeError("unexpected parse error")
        return orig(symbol, *a, **k)

    data.klines = flaky
    run(eng.bars_cycle())
    assert "BBBUSDT" in {p["symbol"] for p in store.get_positions()}
    assert eng.stats["errors"] >= 1


# ------------------------------------------------------------------ reconcile / persistence
def test_restart_restores_paper_state_and_continues(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng, "AAAUSDT")
    eng.persist_state(force=True)
    wallet = eng.ex.wallet
    eng2, _ = mk_engine(store, tmp_path)                           # new process, same DB
    assert eng2.ex.wallet == pytest.approx(wallet)
    assert len(store.get_positions()) == 1 and {o.kind for o in eng2.ex.open_conditional_orders()} == {"SL", "TP"}
    assert eng2.start_equity == pytest.approx(100.0)


def test_reconcile_records_position_that_vanished_while_down(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng, "AAAUSDT")
    p = store.get_positions()[0]
    eng.persist_state(force=True)
    # while the engine is down the paper exchange state loses the position (stop filled)
    st = store.kv_get("paper_state")
    st["pos"] = {}
    st["conds"] = {}
    store.kv_set("paper_state", st)
    eng2, _ = mk_engine(store, tmp_path)
    assert store.get_positions() == [] and store.get_trades(1)[0]["symbol"] == "AAAUSDT"


def test_symbol_dropping_out_of_universe_keeps_its_position(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng, "AAAUSDT")
    eng.universe = ["BBBUSDT"]
    assert "AAAUSDT" in eng.watchlist() and "AAAUSDT" in eng.trade_symbols()
    data.marks["AAAUSDT"] = store.get_positions()[0]["tp_price"] * 1.01
    run(eng.prices_cycle())
    assert store.get_trades(1)[0]["exit_reason"] == "tp"             # ran to TP


def test_equity_and_status_written(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    run(eng.prices_cycle())
    run(eng.heartbeat_cycle())
    run(eng.housekeeping_cycle())
    st = store.kv_get("status")
    assert st["mode"] == "paper_local" and st["equity"] == pytest.approx(100.0) and st["max_positions"] == 4   # fee-aware
    assert st["kill"]["dd_limit_pct"] == 50.0
    assert store.get_equity(5) and store.heartbeat()["data"]["running"] is True
    assert store.kv_get("market")["btc_price"] > 0


# ------------------------------------------------------------------ model registry / shadow / promotion
def write_model(dir_: Path, name: str, status: str | None = None, mtime_offset=0):
    dir_.mkdir(exist_ok=True)
    meta = {"name": name, "version": "1", "created_at": "2026-10-01T00:00:00Z", "thresholds": {"long": 0.7, "short": 0.8}}
    if status:
        meta["status"] = status
    p = dir_ / f"{name}.json"
    p.write_text(json.dumps(meta))
    import os
    t = time.time() + mtime_offset
    os.utime(p, (t, t))


@pytest.fixture
def fake_models(monkeypatch):
    def fake_load(entry):
        sc = FixedScorer(0.9 if entry.name != "weak" else 0.0)
        sc.name = entry.name
        return sc, S.ModelInfo(entry.name, "1", entry.created_at, 0.7, 0.8, entry.status)

    monkeypatch.setattr(E, "load_model", fake_load)


def test_status_resolution_from_json(tmp_path):
    d = tmp_path / "models"
    write_model(d, "old", None, -100)
    write_model(d, "new", None, 0)
    got = {e.name: e.status for e in S.scan_models(d)}
    assert got == {"new": "champion", "old": "retired"}              # newest status-less model is the champion
    write_model(d, "chal", "challenger", 10)
    write_model(d, "champ", "champion", -50)
    got = {e.name: e.status for e in S.scan_models(d)}
    assert got["champ"] == "champion" and got["chal"] == "challenger" and got["new"] == "retired"
    assert {e.name: e.status for e in S.scan_models(d, {"chal": "champion", "champ": "retired"})}["chal"] == "champion"


def test_loader_nullscorer_without_models(tmp_path):
    sc, info = S.load_scorer(tmp_path / "nothing")
    assert isinstance(sc, S.NullScorer) and info is None
    assert sc.score("X", np.zeros((64, 3))).prob_long == 0


def test_challenger_shadow_trades_and_promotion_flow(store, tmp_path, fake_models):
    d = tmp_path / "models"
    write_model(d, "champ", "champion")
    write_model(d, "chal", "challenger")
    eng, data = mk_engine(store, tmp_path)
    assert eng.model_info.name == "champ" and set(eng.challengers) == {"chal"}
    run(eng.bars_cycle())                                          # both models score the bar
    assert eng.shadow.has("chal", "AAAUSDT") and eng.shadow.has("champ", "AAAUSDT")
    # drive prices so the shadow positions hit TP
    for s in ("AAAUSDT", "BBBUSDT"):
        data.marks[s] *= 1.05
    run(eng.prices_cycle())
    st = {s: store.get_shadow_trades(s) for s in ("chal", "champ")}
    assert st["chal"] and st["champ"] and st["chal"][0]["exit_reason"] == "tp"
    assert "shadow_trade" in [e["type"] for e in store.events_after(0)]
    # promotion refused: < 50 shadow trades
    err = run(eng.handle_command("promote_model", {"model": "chal"}))
    assert "promotion refused" in err and "50" in err
    # fabricate a track record: 60 trades, expectancy above the champion's
    t0 = int(time.time()) - 10_000
    for i in range(60):
        store.add_shadow_trade({"model": "chal", "symbol": "A", "side": "long", "entry_time": t0 + i, "exit_time": t0 + i + 1,
                                "entry_price": 1, "exit_price": 1, "qty": 1, "pnl": 0.5 if i % 2 else -0.1, "exit_reason": "tp"})
    for i in range(60):
        store.add_shadow_trade({"model": "champ", "symbol": "A", "side": "long", "entry_time": t0 + i, "exit_time": t0 + i + 1,
                                "entry_price": 1, "exit_price": 1, "qty": 1, "pnl": 0.1 if i % 2 else -0.1, "exit_reason": "tp"})
    # a drift alarm blocks it
    eng.drift.level = 1
    assert "drift" in run(eng.handle_command("promote_model", {"model": "chal"}))
    eng.drift.level = 0
    assert run(eng.handle_command("promote_model", {"model": "chal"})) is None
    assert eng.model_info.name == "chal"
    assert store.kv_get("model_registry")["previous_champion"] == "champ"
    assert dict(eng.model_status_list())["champ"] == "retired"
    # instant rollback
    assert run(eng.handle_command("rollback_model", {})) is None
    assert eng.model_info.name == "champ"
    assert "previous champion is" in run(eng.handle_command("rollback_model", {"model": "ghost"}))
    assert "unknown model" in run(eng.handle_command("promote_model", {"model": "ghost"}))
    assert "model is required" in run(eng.handle_command("promote_model", {}))


def test_promotion_requires_beating_champion(store, tmp_path, fake_models):
    d = tmp_path / "models"
    write_model(d, "champ", "champion")
    write_model(d, "chal", "challenger")
    eng, data = mk_engine(store, tmp_path)
    t0 = int(time.time()) - 10_000
    for m, pnl in (("chal", 0.1), ("champ", 0.3)):
        for i in range(60):
            store.add_shadow_trade({"model": m, "symbol": "A", "side": "long", "entry_time": t0 + i, "exit_time": t0 + i + 1,
                                    "entry_price": 1, "exit_price": 1, "qty": 1, "pnl": pnl, "exit_reason": "tp"})
    err = run(eng.handle_command("promote_model", {"model": "chal"}))
    assert err and "does not beat champion" in err
    assert eng.model_info.name == "champ"


def test_live_dashboard_model_info_in_status(store, tmp_path, fake_models):
    write_model(tmp_path / "models", "champ", "champion")
    eng, _ = mk_engine(store, tmp_path)
    run(eng.heartbeat_cycle())
    m = store.kv_get("status")["model"]
    assert m["name"] == "champ" and m["threshold_long"] == 0.7 and m["trained_at"]
