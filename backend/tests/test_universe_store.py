import threading
import time

import pytest

from conftest import FakeData
from tb import universe
from tb.config import Settings, resolve_mode, LiveNotAllowed, load_secrets, settings_from_rows, trading_base_url
from tb.exchange.base import Premium, SymbolFilters, SymbolInfo, Ticker24h
from tb.store import Store


# ------------------------------------------------------------------ universe
def test_select_retention_buffer():
    ranked = [f"S{i}" for i in range(1, 21)]
    # current member S12 ranks 12 = ceil(1.5*8): stays; S13 (rank 13) is dropped
    out = universe.select(ranked, ["S12", "S13", "S1"], 8)
    assert "S12" in out and "S13" not in out and "S1" in out and len(out) == 8
    assert out == sorted(out, key=ranked.index)
    # no incumbents: plain top 8
    assert universe.select(ranked, [], 8) == ranked[:8]


def test_select_fills_slots_when_incumbents_drop_out():
    ranked = [f"S{i}" for i in range(1, 21)]
    out = universe.select(ranked, ["S2", "S19"], 4)          # S19 is outside the buffer
    assert out == ["S1", "S2", "S3", "S4"]


def test_eligibility_rules():
    cfg = Settings()
    now = int(time.time() * 1000)
    info = SymbolInfo("X", onboard_date_ms=now - 30 * 86_400_000)
    t = Ticker24h("X", 1.0, 100e6, 1.0)
    p = Premium("X", 1.0, 1.0, 0.0001)
    assert universe.eligible(info, t, p, cfg, now)
    assert not universe.eligible(info, Ticker24h("X", 1, 49e6, 1), p, cfg, now)                  # volume
    assert not universe.eligible(SymbolInfo("X", onboard_date_ms=now - 3 * 86_400_000), t, p, cfg, now)   # new listing
    assert not universe.eligible(info, t, Premium("X", 1, 1, 0.0012), cfg, now)                  # funding
    assert not universe.eligible(info, t, Premium("X", 1, 1, -0.0012), cfg, now)
    assert not universe.eligible(SymbolInfo("X", status="BREAK"), t, p, cfg, now)
    assert not universe.eligible(SymbolInfo("X", contract_type="CURRENT_QUARTER"), t, p, cfg, now)
    assert not universe.eligible(SymbolInfo("X", quote_asset="USDC"), t, p, cfg, now)


def test_score_is_atr_times_sqrt_volume():
    assert universe.score(2.0, 400.0) == pytest.approx(40.0)
    c = [universe.Candidate("A", 1e8, 0, 1, 0, 1.0), universe.Candidate("B", 1e8, 0, 1, 0, 3.0),
         universe.Candidate("C", 1e8, 0, 1, 0, None)]
    assert [x.symbol for x in universe.rank_candidates(c)] == ["B", "A"]


def test_refresh_with_fake_exchange():
    d = FakeData(("AAAUSDT", "BBBUSDT", "BTCUSDT"))
    d.exchange_info = lambda: {s: SymbolInfo(s, onboard_date_ms=0, filters=SymbolFilters()) for s in d.marks}
    rows = universe.refresh(d, Settings().merge({"universe_size": 2}), {"BBBUSDT": 123})
    assert len(rows) == 2 and [r["rank"] for r in rows] == [1, 2]
    assert all(r["atr_pct"] and r["score"] > 0 and r["last_price"] for r in rows)
    assert {r["symbol"]: r["in_universe_since"] for r in rows}.get("BBBUSDT", 123) == 123


# ------------------------------------------------------------------ config
def test_settings_merge_validation_and_api_shape():
    s = Settings()
    api = s.to_api()
    assert list(api) == ["slice_margin_usd", "leverage", "max_positions", "max_leverage", "max_total_leverage", "max_margin_ratio", "max_open_risk_pct", "exit_mode", "tp_usd", "sl_usd", "tp_atr",
                         "sl_atr", "trail", "short_enabled", "prob_threshold_long", "prob_threshold_short", "corr_cap",
                         "universe_size", "universe_refresh_s", "kill_dd_pct", "dd_basis", "fee_taker", "sound", "capital_usd", "decision_every_bars",
                             "cooldown_bars", "max_trades_per_day", "entry_mode", "maker_timeout_s", "maker_reprices", "maker_on_timeout"]
    assert s.merge({"trail": {"enabled": True}}).trail.activate_r == 1.0
    for bad in ({"leverage": 0}, {"exit_mode": "x"}, {"prob_threshold_long": 1.5}, {"nope": 1}, {"trail": {"zzz": 1}},
                {"leverage": "abc"}, {"slice_margin_usd": float("nan")}):
        with pytest.raises(ValueError):
            s.merge(bad)
    with pytest.raises(ValueError):
        s.merge({"paper_start_equity": 5}, allow_internal=False)


def test_kill_limit_by_mode():
    s = Settings()
    assert s.kill_limit("paper_local") == 50.0 and s.kill_limit("live") == 25.0 and s.kill_limit("binance_demo") == 25.0
    assert s.merge({"kill_dd_pct": 10}).kill_limit("live") == 10.0


def test_live_needs_env_and_flag():
    with pytest.raises(LiveNotAllowed):
        resolve_mode("live", env={})
    with pytest.raises(LiveNotAllowed):
        resolve_mode("live", env={"ALLOW_LIVE_TRADING": "true"})
    with pytest.raises(LiveNotAllowed):
        resolve_mode("live", live_flag=True, env={})
    assert resolve_mode("live", live_flag=True, env={"ALLOW_LIVE_TRADING": "true"}) == "live"
    assert resolve_mode("live", env={"ALLOW_LIVE_TRADING": "true", "TB_CONFIRM_LIVE": "true"}) == "live"
    assert resolve_mode(None, env={}) == "paper_local"
    with pytest.raises(ValueError):
        resolve_mode("moon", env={})


def test_base_urls():
    assert trading_base_url("live", {}) == "https://fapi.binance.com"
    assert trading_base_url("binance_demo", {}) == "https://demo-fapi.binance.com"
    assert trading_base_url("binance_demo", {"BINANCE_ENV": "testnet"}) == "https://testnet.binancefuture.com"
    assert trading_base_url("paper_local", {}) == "https://fapi.binance.com"


def test_secrets_masked():
    s = load_secrets({"BINANCE_API_KEY": "abcdefghijkl", "BINANCE_API_SECRET": "supersecretvalue",
                      "GOOGLE_CHAT_WEBHOOK_URL": "https://chat.googleapis.com/v1/spaces/X/messages?key=K&token=T"})
    for txt in (repr(s), str(s), f"{s}"):
        assert "abcdefghijkl" not in txt and "supersecretvalue" not in txt and "token=T" not in txt
    assert s.has_binance


def test_settings_from_rows_ignores_garbage():
    s = settings_from_rows({"leverage": 5, "exit_mode": "bogus", "unknown": 1})
    assert s.leverage == 5 and s.exit_mode == "fixed"


# ------------------------------------------------------------------ store
def test_settings_roundtrip(store):
    assert store.get_settings() == Settings()
    new = store.patch_settings({"leverage": 5, "trail": {"enabled": True, "activate_r": 2}})
    again = store.get_settings()
    assert again == new and again.leverage == 5 and again.trail.activate_r == 2
    with pytest.raises(ValueError):
        store.patch_settings({"leverage": 999})
    assert store.get_settings().leverage == 5


def test_wal_mode_and_threads(store):
    assert store.one("PRAGMA journal_mode")[0] == "wal"
    errs = []

    def w(i):
        try:
            s2 = Store(store.path)
            for j in range(50):
                s2.add_event("tick", {"i": i, "j": j})
        except Exception as e:           # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and store.max_event_id() == 200


def test_positions_trades_roundtrip(store):
    pid = store.insert_position({"symbol": "AAAUSDT", "side": "long", "entry_price": 100.0, "qty": 1.6, "notional": 160.0,
                                 "leverage": 8, "margin": 20.0, "sl_price": 93.75, "tp_price": 103.13,
                                 "init_sl_price": 93.75, "opened_at": 1000, "prob": 0.8,
                                 "reasons": [{"feature": "f", "label": "l", "value": 1.0, "contribution": 0.2}]})
    p = store.get_positions()[0]
    assert p["id"] == pid and p["reasons"][0]["feature"] == "f" and p["trail_active"] is False
    store.update_position(pid, trail_active=True, sl_price=95.0, mark_price=101.0)
    assert store.get_position(pid)["trail_active"] is True and store.get_position(pid)["sl_price"] == 95.0
    with pytest.raises(KeyError):
        store.update_position(pid, nope=1)
    tid = store.insert_trade({"position_id": pid, "symbol": "AAAUSDT", "side": "long", "entry_price": 100, "exit_price": 103,
                              "qty": 1.6, "opened_at": 1000, "closed_at": 2000, "pnl": 4.8, "fees": 0.16, "funding": -0.01,
                              "exit_reason": "tp", "prob": 0.8})
    store.insert_trade({"symbol": "BBB", "side": "short", "entry_price": 1, "exit_price": 2, "qty": 1, "opened_at": 1,
                        "closed_at": 3000, "pnl": -1.0, "fees": 0.1, "exit_reason": "sl"})
    store.delete_position(pid)
    assert store.get_positions() == []
    t = store.get_trades(10)
    assert t[0]["exit_reason"] == "sl" and t[1]["id"] == tid
    s = store.trade_stats()
    # trades.pnl is NET; fees and funding are costs; win = net pnl > 0
    assert s["n"] == 2 and s["wins"] == 1 and s["net"] == pytest.approx(4.8 - 1.0)
    assert s["fees"] == pytest.approx(0.26) and s["funding"] == pytest.approx(-0.01)
    assert s["gross"] == pytest.approx(3.8 + 0.26 - 0.01)
    assert store.trade_stats(2500)["n"] == 1


def test_signals_equity_candles_universe_roundtrip(store):
    store.add_signal(10, "A", 0.1, 0.2, "none", [{"feature": "f", "label": "x", "value": 1, "contribution": 0.1}])
    store.add_signal(20, "A", 0.9, 0.0, "long")
    store.add_signal(30, "B", 0.0, 0.0, "none")
    assert [s["t"] for s in store.get_signals("A", 10)] == [10, 20] and len(store.get_signals(None, 10)) == 3
    store.add_equity(1, 100.0)
    store.add_equity(1, 101.0)
    store.add_equity(2, 102.0)
    assert store.get_equity(10) == [{"t": 1, "equity": 101.0}, {"t": 2, "equity": 102.0}]
    store.upsert_candles("A", "5m", [(300, 1, 2, 0.5, 1.5, 10), (600, 1.5, 2, 1, 1.8, 11)])
    store.upsert_candles("A", "5m", [(600, 1.5, 2.5, 1, 2.0, 12)])
    c = store.get_candles("A", "5m", 10)
    assert [x["t"] for x in c] == [300, 600] and c[1]["c"] == 2.0 and c[1]["v"] == 12
    store.replace_universe([{"symbol": "A", "rank": 1, "score": 5.0, "last_price": 1.0, "in_universe_since": 5},
                            {"symbol": "B", "rank": 2, "score": 4.0}])
    store.update_universe_prices({"A": 2.5})
    u = store.get_universe()
    assert [x["symbol"] for x in u] == ["A", "B"] and u[0]["last_price"] == 2.5
    store.add_lab_report("backtest", "t", {"sharpe": 1.0}, [{"t": 1, "equity": 100}])
    assert store.get_lab_reports()[0]["summary"] == {"sharpe": 1.0}


def test_events_commands_logs_kv_heartbeat(store):
    a = store.add_event("tick", {"x": 1})
    b = store.add_event("kill", {"y": 2})
    assert b > a and [e["type"] for e in store.events_after(a)] == ["kill"]
    store.prune_events(-1)
    assert store.max_event_id() >= b and store.events_after(0) == []
    cid = store.add_command("close_position", {"position_id": 3})
    assert store.pending_commands()[0]["payload"] == {"position_id": 3} and store.command_result(cid) is None
    store.finish_command(cid, False, "nope")
    assert store.pending_commands() == [] and store.command_result(cid) == {"ok": False, "error": "nope"}
    store.add_log("info", "hello")
    assert store.get_logs(5)[-1]["msg"] == "hello" and store.get_logs(5)[-1]["level"] == "INFO"
    assert store.kv_get("missing", 7) == 7
    store.kv_set("k", {"a": [1, 2]})
    assert store.kv_get("k") == {"a": [1, 2]} and store.kv_get_many(["k", "z"]) == {"k": {"a": [1, 2]}}
    assert store.heartbeat() is None
    store.beat("paper_local", "2.0.0", 123, {"x": 1})
    store.beat("paper_local", "2.0.0", 123, {"x": 2})
    hb = store.heartbeat()
    assert hb["pid"] == 123 and hb["data"] == {"x": 2} and time.time() - hb["t"] < 5


def test_numpy_values_serialise(store):
    import numpy as np
    store.kv_set("n", {"a": np.float32(1.5), "b": np.arange(3)})
    assert store.kv_get("n") == {"a": 1.5, "b": [0, 1, 2]}
