"""capital_usd: the bot's stake on a much larger exchange wallet (testnet = 5000 USDT, plan = 100)."""
import pytest

from test_engine_core import run
from test_engine_testnet import mk_demo, open_
from tb import capital as cap
from tb.config import Settings

PROD = {"AAAUSDT": 100.0, "BBBUSDT": 100.0, "BTCUSDT": 100.0}


def demo(store, tmp_path, wallet=5000.0, settings=None, **kw):
    return mk_demo(store, tmp_path, prod_marks=PROD, venue_marks=dict(PROD), wallet=wallet, settings=settings, **kw)


def test_default_stake_is_100_and_wallet_is_reported_separately(store, tmp_path):
    assert Settings().capital_usd == 100.0
    eng, data, venue = demo(store, tmp_path, settings={"max_open_risk_pct": 20.0})
    assert eng.equity == pytest.approx(100.0) and eng.wallet_equity == pytest.approx(5000.0)
    assert eng.start_equity == pytest.approx(100.0) and eng.peak_equity == pytest.approx(100.0)
    assert eng.available == pytest.approx(100.0) and eng.cash == pytest.approx(100.0)
    run(eng.heartbeat_cycle())
    st = store.kv_get("status")
    assert st["equity"] == pytest.approx(100.0) and st["wallet_equity"] == pytest.approx(5000.0)
    assert st["capital_usd"] == 100.0 and st["max_positions"] == 1              # capacity from the stake: open-risk cap binds (20% of 100)
    assert st["open_risk_pct"] == 0.0 and st["max_open_risk_pct"] == 20.0


def test_slots_follow_the_stake_not_the_wallet(store, tmp_path):
    eng, data, venue = demo(store, tmp_path)
    eng.universe = ["AAAUSDT", "BBBUSDT"]
    syms = ["AAAUSDT", "BBBUSDT"]
    for s in syms:
        open_(eng, s)
    assert len(store.get_positions()) == 2
    # a wallet of 5000 would allow 5 slots; the stake of 100 leaves 2 more slices
    run(eng.prices_cycle())
    assert eng.available == pytest.approx(100.0 - sum(p["margin"] for p in store.get_positions()), abs=0.5)
    assert eng.available < 66


def test_trading_is_capped_by_the_stake(store, tmp_path):
    eng, data, venue = demo(store, tmp_path, settings=None)
    for s in ("AAAUSDT", "BBBUSDT"):
        open_(eng, s)
    eng.available = 5.0                                  # less than half a slice left of the stake
    before = len(store.get_positions())
    open_(eng, "BTCUSDT")
    assert len(store.get_positions()) == before


def test_equity_is_stake_plus_own_realised_pnl_plus_upnl(store, tmp_path):
    eng, data, venue = demo(store, tmp_path)
    store.insert_trade({"symbol": "A", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": 1,
                        "closed_at": 2, "pnl": 7.5, "fees": 0.1, "exit_reason": "tp"})
    store.insert_trade({"symbol": "A", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": 1,
                        "closed_at": 3, "pnl": -2.0, "fees": 0.1, "exit_reason": "sl"})
    eng._refresh_account()
    assert eng.equity == pytest.approx(105.5)            # 100 + 7.5 - 2.0 (trades.pnl is already net)
    venue.wallet = 5100.0
    eng._refresh_account()
    assert eng.equity == pytest.approx(105.5)            # wallet movement not caused by the bot is ignored


def test_equity_capped_by_real_wallet(store, tmp_path):
    eng, data, venue = demo(store, tmp_path, wallet=60.0)
    assert eng.equity == pytest.approx(60.0)             # stake 100 but only 60 exists


def test_whole_wallet_when_capital_is_null(store, tmp_path):
    eng, data, venue = demo(store, tmp_path, settings={"capital_usd": None})
    assert eng.equity == pytest.approx(5000.0) and eng.available == pytest.approx(5000.0)
    assert store.get_settings().capital_usd is None and store.get_settings().to_api()["capital_usd"] is None


def test_paper_mode_ignores_capital(store, tmp_path):
    from test_engine_core import mk_engine
    eng, _ = mk_engine(store, tmp_path)
    assert eng.equity == pytest.approx(100.0) and eng.cfg.capital_usd == 100.0       # paper uses paper_start_equity
    store.insert_trade({"symbol": "A", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": 1,
                        "closed_at": 2, "pnl": 50.0, "fees": 0.1, "exit_reason": "tp"})
    eng._refresh_account()
    assert eng.equity == pytest.approx(100.0)            # simulated wallet only


def test_drawdown_derating_and_kill_use_stake_equity(store, tmp_path):
    eng, data, venue = demo(store, tmp_path)
    # the bot loses 8 USDT: 8% of the stake (derating starts at 40% of the 25% limit = 10%) -> still x1
    store.insert_trade({"symbol": "A", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": 1,
                        "closed_at": 2, "pnl": -8.0, "fees": 0.1, "exit_reason": "sl"})
    run(eng.prices_cycle())
    assert eng.equity == pytest.approx(92.0) and eng.risk_multiplier()[0] == pytest.approx(1.0)
    store.insert_trade({"symbol": "A", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": 1,
                        "closed_at": 3, "pnl": -9.0, "fees": 0.1, "exit_reason": "sl"})      # 17% -> derated
    run(eng.prices_cycle())
    mult, parts = eng.risk_multiplier()
    assert parts["dd_pct"] == pytest.approx(17.0) and mult == pytest.approx(0.65, abs=0.01)    # 68% of the limit -> 1 - 0.5*(0.28/0.4)
    assert not eng.kill["tripped"]
    store.insert_trade({"symbol": "A", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": 1,
                        "closed_at": 4, "pnl": -9.0, "fees": 0.1, "exit_reason": "sl"})      # 26% >= 25% kill limit
    run(eng.prices_cycle())
    assert eng.kill["tripped"] and "drawdown" in eng.kill["reason"]
    # the wallet itself never moved (5000): the old behaviour would not have tripped at all
    assert eng.wallet_equity == pytest.approx(5000.0)


def test_reset_kill_does_not_reset_capital(store, tmp_path):
    eng, data, venue = demo(store, tmp_path)
    store.insert_trade({"symbol": "A", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": 1,
                        "closed_at": 2, "pnl": -26.0, "fees": 0.1, "exit_reason": "sl"})
    run(eng.prices_cycle())
    assert eng.kill["tripped"]
    assert run(eng.handle_command("reset_kill", {})) is None
    assert not eng.kill["tripped"] and eng.peak_equity == pytest.approx(74.0)       # re-baselined peak ...
    assert eng.equity == pytest.approx(74.0)                                          # ... but the stake keeps the loss
    assert cap.floor_id(store) == 0


def test_restart_keeps_the_starting_point(store, tmp_path):
    eng, data, venue = demo(store, tmp_path)
    store.insert_trade({"symbol": "A", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": 1,
                        "closed_at": 2, "pnl": 5.0, "fees": 0.1, "exit_reason": "tp"})
    run(eng.prices_cycle())
    eng.persist_state(force=True)
    start, peak = eng.start_equity, eng.peak_equity
    eng2, _, _ = demo(store, tmp_path)                       # "restart": same DB
    assert eng2.equity == pytest.approx(105.0)
    assert eng2.start_equity == pytest.approx(start) and eng2.peak_equity == pytest.approx(peak)
    assert not eng2.basis_fresh


def test_first_run_after_upgrade_rebases_old_wallet_sized_peak(store, tmp_path):
    # state written by the previous engine version: peak/start on the whole-wallet basis, a drawdown kill on it
    store.kv_set("peak_equity", 5000.0)
    store.kv_set("start_equity", 5000.0)
    store.kv_set("kill", {"tripped": True, "reason": "drawdown 98.0% >= 25% of peak", "t": 1})
    eng, data, venue = demo(store, tmp_path)
    assert eng.equity == pytest.approx(100.0) and eng.peak_equity == pytest.approx(100.0) and eng.start_equity == pytest.approx(100.0)
    assert not eng.kill["tripped"]
    assert store.kv_get(cap.BASIS_KEY) == 100.0
    # a manual kill is NOT silently cleared
    store2_kill = {"tripped": True, "reason": "manual kill", "t": 1}
    store.kv_set("kill", store2_kill)
    eng3, _, _ = demo(store, tmp_path)
    assert eng3.kill["tripped"]


def test_changing_capital_rebases_peak(store, tmp_path):
    eng, data, venue = demo(store, tmp_path)
    store.patch_settings({"capital_usd": 50})
    run(eng.heartbeat_cycle())
    assert eng.equity == pytest.approx(50.0) and eng.peak_equity == pytest.approx(50.0)
    assert not eng.kill["tripped"]
    store.patch_settings({"capital_usd": None})
    run(eng.heartbeat_cycle())
    assert eng.equity == pytest.approx(5000.0) and eng.peak_equity == pytest.approx(5000.0)


def test_settings_validation(store):
    assert store.patch_settings({"capital_usd": 250}).capital_usd == 250.0
    assert store.patch_settings({"capital_usd": None}).capital_usd is None
    for bad in (0, -5, "abc"):
        with pytest.raises(ValueError):
            store.patch_settings({"capital_usd": bad})


def test_api_status_reports_both_equities(store, tmp_path):
    from fastapi.testclient import TestClient
    from tb.api import create_app
    eng, data, venue = demo(store, tmp_path)
    run(eng.heartbeat_cycle())
    c = TestClient(create_app(store.path, ui=tmp_path / "noui"))
    st = c.get("/api/status").json()
    assert st["equity"] == pytest.approx(100.0) and st["wallet_equity"] == pytest.approx(5000.0)
    assert st["drawdown_pct"] == 0.0
    assert c.get("/api/settings").json()["capital_usd"] == 100.0
    assert c.put("/api/settings", json={"capital_usd": 200}).json()["capital_usd"] == 200.0
    assert c.put("/api/settings", json={"capital_usd": 0}).status_code == 422


def test_restart_reconciles_open_positions_and_keeps_exchange_stops(store, tmp_path):
    eng, data, venue = demo(store, tmp_path)
    open_(eng, "AAAUSDT")
    assert len(store.get_positions()) == 1 and len(venue.orders) == 2
    eng.persist_state(force=True)
    orders_before = [o.order_id for o in venue.orders]
    # restart: new engine process, same DB and same exchange state
    from tb import engine as E
    from tb.alerts import AlertDispatcher
    eng2 = E.Engine(store, data, venue, "binance_demo", AlertDispatcher([], min_interval=0), models_path=tmp_path / "models",
                    clock=eng.clock)
    eng2.startup()
    assert len(store.get_positions()) == 1                                           # still tracked, not abandoned
    assert [o.order_id for o in venue.orders] == orders_before                        # exchange stops untouched
    assert not [x for x in venue.log if x[0] == "market" and x[4]]                    # nothing was closed
    # a position that exists on the exchange but not in the DB is adopted when it has a stop ...
    store.execute("DELETE FROM positions")
    eng3 = E.Engine(store, data, venue, "binance_demo", AlertDispatcher([], min_interval=0), models_path=tmp_path / "models",
                    clock=eng.clock)
    eng3.startup()
    adopted = store.get_positions()
    assert len(adopted) == 1 and adopted[0]["symbol"] == "AAAUSDT" and adopted[0]["sl_order_id"] in orders_before
    # ... and a position that closed while the engine was down is recorded as a trade
    venue.pos.clear()
    venue.orders.clear()
    eng4 = E.Engine(store, data, venue, "binance_demo", AlertDispatcher([], min_interval=0), models_path=tmp_path / "models",
                    clock=eng.clock)
    eng4.startup()
    assert store.get_positions() == [] and len(store.get_trades(5)) == 1


# ------------------------------------------------------------------ capacity rules (leverage / margin ratio / open risk)
def test_open_risk_cap_blocks_the_nth_correlated_trade(store, tmp_path, caplog):
    import logging
    syms = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "BTCUSDT")
    eng, data, venue = mk_demo(store, tmp_path, prod_marks={s: 100.0 for s in syms}, venue_marks={s: 100.0 for s in syms},
                               settings={"max_open_risk_pct": 25.0, "corr_cap": {"max_corr": 0.99, "max_correlated": 9}},
                               wallet=100.0)
    eng.cfg = store.get_settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0}, allow_internal=True)
    eng.universe = list(syms[:4])
    with caplog.at_level(logging.INFO, logger="tb.engine"):
        for s in syms[:4]:
            open_(eng, s)
    # each stop-out risks ~10.1 USD (+fees): two fit under 25% of 100, the third would be 30%
    assert len(store.get_positions()) == 2
    assert any("blocked by rule open_risk" in r.message for r in caplog.records)
    cap = eng.capacity()
    assert 20 < cap["open_risk_pct"] <= 25 and cap["max_open_risk_pct"] == 25.0


def test_total_leverage_rule_blocks_entry(store, tmp_path, caplog):
    import logging
    eng, data, venue = mk_demo(store, tmp_path, prod_marks=PROD, venue_marks=dict(PROD), wallet=100.0,
                               settings={"max_open_risk_pct": 100.0, "max_total_leverage": 1.5})
    eng.cfg = store.get_settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0}, allow_internal=True)
    with caplog.at_level(logging.INFO, logger="tb.engine"):
        open_(eng, "AAAUSDT")                  # 160 notional on 100 equity = 1.6x > 1.5x
    assert store.get_positions() == []
    assert any("blocked by rule total_leverage" in r.message for r in caplog.records)


def test_high_mmr_coin_blocked_by_margin_ratio(store, tmp_path, caplog):
    import logging
    from tb.risk import Bracket
    eng, data, venue = mk_demo(store, tmp_path, prod_marks=PROD, venue_marks=dict(PROD), wallet=100.0,
                               settings={"max_open_risk_pct": 100.0, "max_margin_ratio": 0.05})
    eng.cfg = store.get_settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0}, allow_internal=True)
    venue.leverage_brackets = lambda s: [Bracket(0, 1e9, 0.5 if s == "BBBUSDT" else 0.01, 0.0, 20)]
    with caplog.at_level(logging.INFO, logger="tb.engine"):
        open_(eng, "BBBUSDT")                  # mmr 50%: MM 80 USDT on 100 equity -> far above 5%
        open_(eng, "AAAUSDT")                  # mmr 1%:  MM 1.6 -> 1.6% ok
    assert [p["symbol"] for p in store.get_positions()] == ["AAAUSDT"]
    assert any("blocked by rule margin_ratio" in r.message or "liquidation" in r.message for r in caplog.records)


def test_leverage_is_set_on_the_exchange_before_entry_and_recorded(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks=PROD, venue_marks=dict(PROD), wallet=100.0,
                               settings={"max_open_risk_pct": 100.0})
    open_(eng, "AAAUSDT")
    kinds = [x[0] for x in venue.log]
    assert kinds.index("iso") < kinds.index("lev") < kinds.index("market")
    lev = [x for x in venue.log if x[0] == "lev"][0]
    p = store.get_positions()[0]
    assert lev[2] == p["leverage"] == 9
    assert p["margin"] == pytest.approx(p["notional"] / 9, rel=0.01)
    # liquidation of the chosen leverage is >= 1.5x the stop distance away
    from tb.risk import liq_price
    liq = liq_price("long", p["entry_price"], p["leverage"], eng.cfg.maint_margin_rate)
    assert abs(p["entry_price"] - liq) >= 1.5 * abs(p["entry_price"] - p["sl_price"]) * 0.999


def test_leverage_falls_back_when_the_exchange_rejects_it(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks=PROD, venue_marks=dict(PROD), wallet=100.0,
                               settings={"max_open_risk_pct": 100.0})
    from tb.exchange.base import ExchangeError
    orig = venue.set_leverage

    def picky(s, lev):
        if lev > 5:
            raise ExchangeError("Leverage is not valid", code=-4028)
        orig(s, lev)

    venue.set_leverage = picky
    open_(eng, "AAAUSDT")
    p = store.get_positions()[0]
    assert p["leverage"] == 4 and p["margin"] == pytest.approx(p["notional"] / 4, rel=0.01)     # 9 -> 9//2 = 4


def test_capacity_on_a_200_usdt_stake(store, tmp_path):
    eng, data, venue = mk_demo(store, tmp_path, prod_marks=PROD, venue_marks=dict(PROD), wallet=5000.0,
                               settings={"capital_usd": 200.0, "max_open_risk_pct": 20.0})
    run(eng.heartbeat_cycle())
    st = store.kv_get("status")
    assert st["equity"] == pytest.approx(200.0)
    assert st["max_positions"] == 3                              # open risk 20% of 200 = 40 USDT / ~10.3 per stop-out
    loose = store.patch_settings({"max_open_risk_pct": 100.0}, allow_internal=True)
    eng.cfg = loose
    assert eng.capacity()["max_positions"] == 11                 # then free margin binds (200 / 17.9)
