"""Safety layer: debouncing, plausibility, drift, merged state, guard process logic, shadow/promotion."""
import numpy as np
import pytest

from tb import guard as G
from tb.config import Settings
from tb.exchange.base import (Balance, ConditionalOrder, ExchangeError, NetworkError, PositionInfo, SymbolFilters)
from tb.safety import (DEGRADED, FLATTEN, HALTED, NORMAL, Debouncer, DriftMonitor, FaultTracker, check_bars, check_tick,
                       drift_z, merged_safety, slippage_drift, worst)
from tb.shadow import Registry, ShadowBook, promotion_check, shadow_stats, vs_champion

BAR = 300_000


# ------------------------------------------------------------------ debounce / hysteresis
def test_fault_matures_only_after_n_consecutive_bad():
    d = Debouncer(3, 5)
    assert [d.observe("k", True) for _ in range(2)] == [False, False]
    assert d.observe("k", False) is False             # a good cycle resets the streak
    assert [d.observe("k", True) for _ in range(3)] == [False, False, True]
    assert d.active("k")


def test_fault_clears_only_after_m_consecutive_good():
    d = Debouncer(3, 5)
    for _ in range(3):
        d.observe("k", True)
    assert d.active("k")
    for i in range(4):
        assert d.observe("k", False) is True          # hysteresis: still active after 1..4 good cycles
    assert d.observe("k", True) is True               # a bad cycle restarts the recovery count
    for _ in range(4):
        assert d.observe("k", False) is True
    assert d.observe("k", False) is False             # 5th consecutive good clears it


def test_fault_tracker_symbol_exclusion_and_exchange_escalation():
    ft = FaultTracker(3, 5)
    for _ in range(2):
        ft.observe_symbol("A", ["big_move"])
    assert not ft.excluded("A") and ft.warnings() == [("A", "big_move", 2)]
    ft.observe_symbol("A", ["big_move"])
    assert ft.excluded("A") and not ft.excluded("B")
    f = ft.faults()
    assert f[0]["symbol"] == "A" and f[0]["kind"] == "big_move" and f[0]["since"] is not None
    for _ in range(5):
        ft.observe_symbol("A", [], ("big_move",))
    assert not ft.excluded("A") and ft.faults() == []
    assert [ft.observe_exchange(True) for _ in range(3)] == [False, False, True] and ft.exchange_fault()


# ------------------------------------------------------------------ plausibility
def _bars(n=30, price=100.0, vol=10.0, now=None):
    now = now or 1_800_000_000_000
    ot = now // BAR * BAR - np.arange(n - 1, -1, -1) * BAR
    c = np.full(n, price, dtype=float)
    return dict(open_time=ot, o=c.copy(), h=c * 1.001, l=c * 0.999, c=c, v=np.full(n, vol), now_ms=int(now))


def chk(b, atr=0.3, **kw):
    return check_bars(b["open_time"], b["o"], b["h"], b["l"], b["c"], b["v"], atr, b["now_ms"], **kw)


def test_plausible_bars_pass():
    assert chk(_bars()) == []


def test_big_move_uses_max_of_15pct_and_8_atr():
    b = _bars()
    b["c"][-1] = 112.0
    b["h"][-1] = 113.0
    assert chk(b) == []                                 # 12% < 15%
    b["c"][-1] = 120.0
    b["h"][-1] = 121.0
    assert "big_move" in chk(b)                         # 20% > max(15%, 8*0.3%)
    assert "big_move" not in chk(b, atr=3.0)            # 8*ATR = 24% of price: legitimately volatile coin


def test_bad_prices_high_low_zero_volume_and_stale():
    b = _bars()
    b["h"][-2] = 90.0                                    # high < low
    assert "high_low" in chk(b)
    b = _bars()
    b["c"][-1] = -1.0
    assert "bad_price" in chk(b)
    b = _bars()
    b["c"][-1] = np.nan
    assert "bad_price" in chk(b)
    b = _bars()
    b["v"][-2] = 0.0                                     # last closed bar (the last one is still forming)
    b["now_ms"] = int(b["open_time"][-1]) + 10_000
    assert "zero_volume" in chk(b)
    b = _bars()
    b["now_ms"] = int(b["open_time"][-1]) + 4 * BAR      # newest bar more than 2 bars behind
    assert "stale" in chk(b)
    b = _bars()
    b["now_ms"] = int(b["open_time"][-1]) + BAR + 1000
    assert "stale" not in chk(b)


def test_tick_checks():
    assert check_tick(100.0, 100.5) == []
    assert check_tick(100.0, 102.0) == ["mark_last_diff"]
    assert check_tick(-1.0, 100.0) == ["bad_price"]
    assert check_tick(float("nan"), None) == ["bad_price"]
    assert check_tick(100.0, None) == []


# ------------------------------------------------------------------ drift
def test_drift_z_formula():
    probs = [0.6] * 30
    wins = [True] * 18 + [False] * 12                    # realised 0.6 == predicted
    assert drift_z(probs, wins) == pytest.approx(0.0)
    wins = [True] * 12 + [False] * 18                    # realised 0.4
    assert drift_z(probs, wins) == pytest.approx((0.4 - 0.6) / np.sqrt(0.6 * 0.4 / 30))
    assert drift_z([], []) is None


def _mon_feed(mon, p, win_rate, n=30):
    k = round(win_rate * n)
    return mon.update([p] * n, [True] * k + [False] * (n - k))


def test_drift_thresholds_and_hysteresis():
    m = DriftMonitor(30)
    assert m.update([0.6] * 10, [True] * 10).level == 0                      # not enough trades yet
    st = _mon_feed(m, 0.6, 0.6)
    assert st.level == 0 and st.mult == 1.0
    st = _mon_feed(m, 0.6, 0.40)                                              # z ~ -2.2 -> size x0.5
    assert -3 < st.z < -2 and st.level == 1 and st.mult == 0.5 and not m.entries_blocked()
    st = _mon_feed(m, 0.6, 0.30)                                              # z ~ -3.3 -> block entries
    assert st.z < -3 and st.level == 2 and m.entries_blocked()
    st = _mon_feed(m, 0.6, 0.50)                                              # z ~ -1.1: not yet recovered
    assert st.level == 2 and -2 < st.z < -1
    st = _mon_feed(m, 0.6, 0.57)                                              # z ~ -0.3 > -1: cleared
    assert st.z > -1 and st.level == 0 and st.mult == 1.0


def test_drift_uses_only_last_n():
    m = DriftMonitor(30)
    probs = [0.6] * 100
    wins = [False] * 70 + [True] * 18 + [False] * 12      # bad history, last 30 are on target
    assert m.update(probs, wins).level == 0


def test_slippage_drift():
    assert not slippage_drift([0.02] * 10, 0.02)
    assert slippage_drift([0.05] * 10, 0.02, 2.0)
    assert not slippage_drift([0.5] * 3, 0.02)            # too few samples


def test_worst_and_merge():
    assert worst(NORMAL, DEGRADED) == DEGRADED and worst(FLATTEN, HALTED, NORMAL) == HALTED
    m = merged_safety({"state": DEGRADED, "reasons": ["a"]}, {"state": NORMAL, "reasons": []})
    assert m["state"] == DEGRADED and m["reasons"] == ["a"] and set(m) == {"state", "reasons", "derate_mult", "drift", "faults"}
    m = merged_safety(None, {"state": NORMAL}, guard_age_s=None, guard_required=True)
    assert m["state"] == DEGRADED and "guard" in m["reasons"][0]
    m = merged_safety(None, None, guard_age_s=None, guard_required=False)
    assert m["state"] == NORMAL


# ------------------------------------------------------------------ guard
class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeTrade:
    def __init__(self):
        self.pos = []
        self.orders = []
        self.equity = 100.0
        self.fail_list = False
        self.fail_stop = False
        self.placed = []
        self.closed = []

    def filters(self, s):
        return SymbolFilters(0.01, 0.001, 0.001, 5)

    def positions(self):
        if self.fail_list:
            raise NetworkError("down")
        return list(self.pos)

    def open_conditional_orders(self, symbol=None):
        if self.fail_list:
            raise NetworkError("down")
        return list(self.orders)

    def balance(self):
        return Balance(self.equity, self.equity, 0.0)

    def place_stop(self, sym, side, price, qty):
        if self.fail_stop:
            raise ExchangeError("would trigger", code=-2021)
        self.placed.append((sym, side, price, qty))
        self.orders.append(ConditionalOrder("o:1", sym, "SL", side, price, qty, True))
        return "o:1"

    def market_order(self, sym, side, qty, reduce_only=False):
        self.closed.append((sym, side, qty))
        self.pos = [p for p in self.pos if p.symbol != sym]

    def cancel_all(self, sym):
        self.orders = [o for o in self.orders if o.symbol != sym]

    def cancel_order(self, sym, oid):
        self.orders = [o for o in self.orders if o.order_id != oid]


def pos_row(store, sym="AAAUSDT", sl=93.75):
    return store.insert_position({"symbol": sym, "side": "long", "entry_price": 100.0, "qty": 1.6, "notional": 160.0,
                                  "leverage": 8, "margin": 20.0, "sl_price": sl, "tp_price": 103.0,
                                  "init_sl_price": sl, "opened_at": 1, "sl_order_id": "o:9"})


def xpos(sym="AAAUSDT"):
    return PositionInfo(sym, "long", 1.6, 100.0, 100.0, 0.0, 8, 88.0, 20.0)


def test_guard_degraded_on_stale_heartbeat_with_positions_and_recovers(store):
    clk = Clock()
    ex = FakeTrade()
    ex.pos = [xpos()]
    ex.orders = [ConditionalOrder("o:9", "AAAUSDT", "SL", "SELL", 93.75, 1.6, True)]
    pos_row(store)
    g = G.Guard(store, "binance_demo", ex, clock=clk)
    store.beat("binance_demo", "2", 1)
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    assert g.step()["state"] == NORMAL
    clk.t += 45                                              # heartbeat now 45 s old (> 30 s)
    r = g.step()
    assert r["state"] == DEGRADED and "heartbeat stale" in r["reasons"][0]
    assert store.kv_get("safety_guard")["state"] == DEGRADED
    assert store.kv_get("guard_heartbeat")["t"] == clk.t
    evs = [e["data"]["state"] for e in store.events_after(0) if e["type"] == "safety"]
    assert evs == [NORMAL, DEGRADED]                          # one event per state change
    # engine comes back: hysteresis, needs fault_recover_m (5) good cycles
    for i in range(4):
        store.execute("UPDATE heartbeat SET t=?", (clk.t,))
        assert g.step()["state"] == DEGRADED, i
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    assert g.step()["state"] == NORMAL


def test_guard_stale_heartbeat_without_positions_is_normal(store):
    clk = Clock()
    g = G.Guard(store, "binance_demo", FakeTrade(), clock=clk)
    assert g.step()["state"] == NORMAL


def test_guard_places_missing_stop_itself_when_engine_stale(store):
    clk = Clock()
    ex = FakeTrade()
    ex.pos = [xpos()]
    pos_row(store, sl=93.75)
    store.beat("binance_demo", "2", 1)
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    clk.t += 100                                             # engine stale -> guard acts immediately
    g = G.Guard(store, "binance_demo", ex, clock=clk)
    r = g.step()
    assert r["state"] == DEGRADED
    assert ex.placed == [("AAAUSDT", "SELL", 93.75, 1.6)]
    assert store.get_positions()[0]["sl_order_id"] == "o:1"


def test_guard_waits_two_cycles_before_touching_stops_when_engine_alive(store):
    clk = Clock()
    ex = FakeTrade()
    ex.pos = [xpos()]
    pos_row(store)
    g = G.Guard(store, "binance_demo", ex, clock=clk)
    store.beat("binance_demo", "2", 1)
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    assert g.step()["state"] == NORMAL and ex.placed == []   # first sighting: the engine may be mid-trail
    assert g.step()["state"] == DEGRADED and len(ex.placed) == 1


def test_guard_flattens_when_stop_cannot_be_placed(store):
    clk = Clock()
    ex = FakeTrade()
    ex.pos = [xpos()]
    ex.fail_stop = True
    pos_row(store)
    store.beat("binance_demo", "2", 1)
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    clk.t += 100
    g = G.Guard(store, "binance_demo", ex, clock=clk)
    r = g.step()
    assert ex.closed == [("AAAUSDT", "SELL", 1.6)]
    # per-symbol: only THAT coin was closed; the account is NOT flattened/halted (no new entries while the issue shows)
    assert r["state"] == DEGRADED and "stop missing on AAAUSDT" in r["reasons"]


def test_guard_judges_the_bot_stake_not_the_wallet(store):
    """A 5000 USDT wallet with a 100 USDT stake: wallet noise is irrelevant, the stake's drawdown trips the guard."""
    clk = Clock()
    ex = FakeTrade()
    ex.equity = 5000.0
    store.kv_set("peak_equity", 100.0)
    store.kv_set("capital_floor_trade_id", 0)
    g = G.Guard(store, "binance_demo", ex, clock=clk)
    store.beat("binance_demo", "2", 1)
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    assert g.step()["state"] == NORMAL                       # bot equity = min(100, 5000) = 100
    ex.equity = 4990.0                                       # wallet -10 USDT is only -0.2% of the wallet ...
    store.insert_trade({"symbol": "A", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": 1,
                        "closed_at": 2, "pnl": -26.0, "fees": 0.1, "exit_reason": "sl"})     # ... but -26% of the stake
    r = g.step()
    assert r["state"] == HALTED and "drawdown" in " ".join(r["reasons"])


def test_guard_ignores_drawdown_until_stake_initialised(store):
    clk = Clock()
    ex = FakeTrade()
    ex.equity = 40.0
    store.kv_set("peak_equity", 100.0)
    g = G.Guard(store, "binance_demo", ex, clock=clk)
    store.beat("binance_demo", "2", 1)
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    assert g.step()["state"] == NORMAL


def test_guard_drawdown_flatten_then_halted_and_manual_reset(store):
    clk = Clock()
    ex = FakeTrade()
    ex.pos = [xpos()]
    ex.orders = [ConditionalOrder("o:9", "AAAUSDT", "SL", "SELL", 93.75, 1.6, True)]
    pos_row(store)
    store.kv_set("peak_equity", 100.0)
    store.kv_set("capital_floor_trade_id", 0)               # stake initialised by the engine (capital_usd=100)
    store.beat("binance_demo", "2", 1)
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    g = G.Guard(store, "binance_demo", ex, clock=clk)
    assert g.step()["state"] == NORMAL
    ex.equity = 74.0                                         # -26% >= 25% live/demo kill limit
    r = g.step()
    assert ex.closed and r["state"] == HALTED and "drawdown" in " ".join(r["reasons"])
    store.delete_position(1)
    ex.equity = 100.0                                        # even if equity recovers, HALTED is sticky
    assert g.step()["state"] == HALTED
    store.kv_set(G.KEY_RESET, clk.t + 1)                     # engine's reset_kill
    clk.t += 5
    assert g.step()["state"] == NORMAL


def test_guard_exchange_fault_debounced_then_degraded_then_flatten_if_stops_unconfirmable(store):
    clk = Clock()
    ex = FakeTrade()
    ex.pos = [xpos()]
    pos_row(store)
    store.beat("binance_demo", "2", 1)
    g = G.Guard(store, "binance_demo", ex, clock=clk, flatten_after_s=60)
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    g.step()
    ex.fail_list = True
    states = []
    for _ in range(3):
        clk.t += 5
        store.execute("UPDATE heartbeat SET t=?", (clk.t,))
        states.append(g.step()["state"])
    assert states == [NORMAL, NORMAL, DEGRADED]              # matured after 3 consecutive failures
    clk.t += 70
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    assert g.step()["state"] == FLATTEN                      # still cannot confirm stops after the grace period
    ex.fail_list = False
    ex.orders = []
    assert g.step()["state"] in (HALTED, FLATTEN)


def test_guard_paper_acts_through_commands_table(store):
    clk = Clock()
    g = G.Guard(store, "paper_local", None, clock=clk)
    pos_row(store)
    store.beat("paper_local", "2", 1)
    store.execute("UPDATE heartbeat SET t=?", (clk.t,))
    store.kv_set("paper_state", {"conds": {}})              # position without a simulated SL
    store.kv_set("status", {"equity": 100.0, "kill": {"tripped": False}})
    g.step()
    assert g.step()["state"] == DEGRADED
    cmds = store.pending_commands()
    assert [c["action"] for c in cmds] == ["close_position"] and cmds[0]["payload"]["position_id"] == 1
    # drawdown beyond the paper kill limit (50%) -> kill command
    store.execute("DELETE FROM commands")
    store.kv_set("paper_state", {"conds": {"p:1": {"symbol": "AAAUSDT", "kind": "SL"}}})
    store.kv_set("peak_equity", 100.0)
    store.kv_set("status", {"equity": 40.0, "kill": {"tripped": False}})
    g2 = G.Guard(store, "paper_local", None, clock=clk)
    r = g2.step()
    assert r["state"] == FLATTEN and [c["action"] for c in store.pending_commands()] == ["kill"]


def test_guard_halts_when_derated_below_min_notional_zombie_gap(store):
    """dd just below the kill limit: derating leaves no tradeable size, yet the kill never trips."""
    clk = Clock()
    g = G.Guard(store, "paper_local", None, clock=clk)
    store.kv_set("peak_equity", 100.0)
    store.kv_set("min_notional_floor", 5.0)
    store.kv_set("status", {"equity": 51.0, "kill": {"tripped": False}})       # 49% dd of a 50% limit: 0.1x -> 16 USD
    assert g.step()["state"] == NORMAL
    store.kv_set("status", {"equity": 50.05, "kill": {"tripped": False}})      # 49.95%: 0.0125x -> 2 USD < 5
    r = g.step()
    assert r["state"] == HALTED and "derated below min notional" in " ".join(r["reasons"])
    assert store.kv_get("safety_guard")["state"] == HALTED
    # sticky until the manual reset_kill (engine re-baselines the peak and signals the guard)
    store.kv_set("status", {"equity": 80.0, "kill": {"tripped": False}})
    assert g.step()["state"] == HALTED
    store.kv_set("peak_equity", 80.0)
    store.kv_set(G.KEY_RESET, clk.t + 1)
    clk.t += 5
    assert g.step()["state"] == NORMAL
    # a symbol with a tiny minimum notional can still trade -> no halt
    store.kv_set("min_notional_floor", 0.1)
    store.kv_set("peak_equity", 100.0)
    store.kv_set("status", {"equity": 50.05, "kill": {"tripped": False}})
    assert G.Guard(store, "paper_local", None, clock=clk).step()["state"] == NORMAL


def test_guard_mirrors_engine_kill_as_halted(store):
    clk = Clock()
    g = G.Guard(store, "paper_local", None, clock=clk)
    store.kv_set("status", {"equity": 60.0, "kill": {"tripped": True, "reason": "dd"}})
    assert g.step()["state"] == HALTED


# ------------------------------------------------------------------ shadow book / promotion
F = SymbolFilters(0.01, 0.001, 0.001, 5.0)


def test_shadow_trade_tp_and_sl_first():
    cfg = Settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0}, allow_internal=True)
    b = ShadowBook(cfg)
    assert b.open("m", "A", "long", 0.8, 100.0, 1000.0, None, F)
    assert not b.open("m", "A", "long", 0.8, 100.0, 1000.0, None, F)           # one per (model, symbol)
    tr = b.on_price("A", 103.2, 1100.0)
    assert len(tr) == 1 and tr[0]["exit_reason"] == "tp"
    assert tr[0]["pnl"] == pytest.approx(5.008 - 1.6 * 100 * 0.0005 - 1.6 * 103.13 * 0.0005, abs=1e-3)
    b.open("m", "A", "long", 0.8, 100.0, 2000.0, None, F)
    # bar wicks both ways -> SL first; bars that opened before entry are ignored
    assert b.on_bar("A", 1_999_000, 90.0, 110.0) == []
    tr = b.on_bar("A", 2_300_000, 90.0, 110.0)
    assert tr[0]["exit_reason"] == "sl" and tr[0]["pnl"] < -9.9


def test_shadow_state_roundtrip():
    b = ShadowBook(Settings())
    b.open("m", "A", "short", 0.9, 100.0, 5.0, None, F)
    c = ShadowBook(Settings())
    c.import_state(b.export_state())
    assert c.has("m", "A") and c.pos["m|A"].side == "short"


def test_promotion_criteria():
    mk = lambda n, e: {"n_trades": n, "expectancy": e, "win_rate": 0.5, "pnl": n * e}
    assert promotion_check(mk(60, 0.2), mk(60, 0.1), 0, 50) == (True, "ok")
    ok, why = promotion_check(mk(49, 0.5), mk(60, 0.1), 0, 50)
    assert not ok and "50" in why                                               # not enough shadow trades
    ok, why = promotion_check(mk(60, 0.1), mk(60, 0.1), 0, 50)
    assert not ok and "expectancy" in why                                       # must be strictly higher
    ok, why = promotion_check(mk(60, 0.05), mk(60, 0.1), 0, 50)
    assert not ok
    ok, why = promotion_check(mk(60, 0.5), mk(60, 0.1), 1, 50)
    assert not ok and "drift" in why                                            # drift alarm blocks it


def test_shadow_stats_and_vs_champion():
    tr = [{"pnl": 1.0}, {"pnl": -0.5}, {"pnl": 2.0}, {"pnl": -1.0}]
    s = shadow_stats(tr)
    assert s["n_trades"] == 4 and s["win_rate"] == 0.5 and s["pnl"] == 1.5 and s["expectancy"] == 0.375
    assert vs_champion(s, {"expectancy": 0.1}) == pytest.approx(0.275)
    assert vs_champion(shadow_stats([]), {"expectancy": 0.1}) is None


def test_guard_is_independent_of_the_engine():
    import subprocess
    import sys
    code = "import sys, tb.guard; bad = [m for m in sys.modules if m in ('tb.engine', 'tb.api')]; sys.exit(1 if bad else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_risk_module_is_import_light():
    """The backtester imports tb.risk: it must not pull in the exchange clients / engine."""
    import subprocess
    import sys
    code = ("import sys, tb.risk; bad = [m for m in sys.modules if m.startswith(('tb.engine', 'tb.exchange', 'tb.api'))];"
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_registry_promote_and_rollback():
    r = Registry({}, None).promote("new", "old")
    assert r.overrides == {"old": "retired", "new": "champion"} and r.previous_champion == "old"
    back = r.rollback("new")
    assert back.overrides == {"old": "champion", "new": "retired"} and back.previous_champion == "new"
    with pytest.raises(ValueError):
        Registry({}, None).rollback("x")
    assert Registry.from_dict(r.to_dict()) == r
