import threading
import time

import pytest
from fastapi.testclient import TestClient

from tb.api import create_app
from tb.store import Store

STATUS_KEYS = {"mode", "running", "equity", "wallet_equity", "start_equity", "peak_equity", "cash", "realized_pnl", "unrealized_pnl",
               "drawdown_pct", "open_positions", "max_positions", "trades_today", "win_rate", "kill", "model",
               "universe_updated_at", "safety", "paused_symbols", "total_leverage", "max_total_leverage", "margin_ratio",
               "max_margin_ratio", "open_risk_pct", "max_open_risk_pct", "models", "decisions", "maker", "xsec_legs", "dd_basis"}


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "tb.sqlite"
    s = Store(db)
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<html>ui</html>")
    app = create_app(db, control_wait_s=0.0, ui=ui)
    c = TestClient(app)
    yield c, s
    s.close()


def seed(s: Store):
    now = int(time.time())
    s.beat("paper_local", "2.0.0", 1)
    s.kv_set("status", {"mode": "paper_local", "running": True, "equity": 104.0, "start_equity": 100.0,
                        "peak_equity": 110.0, "cash": 100.0, "unrealized_pnl": 4.0, "max_positions": 5,
                        "kill": {"tripped": False, "reason": "", "dd_limit_pct": 50.0},
                        "model": {"name": "m1", "version": "3", "trained_at": 1700000000, "threshold_long": 0.7,
                                  "threshold_short": 0.8},
                        "universe_updated_at": now - 60})
    s.kv_set("market", {"btc_price": 80000.0, "btc_change_24h_pct": -0.2, "btc_trend": "down", "market_volatility_pct": 2.0,
                        "avg_funding_rate": 0.0001, "long_short_ratio": 1.5, "updated_at": now})
    s.replace_universe([{"symbol": "ETHUSDT", "rank": 1, "score": 5.0, "atr_pct": 0.3, "quote_volume_24h": 1e9,
                         "change_24h_pct": 1.0, "funding_rate": 0.0001, "open_interest_usd": 1e8, "oi_change_1h_pct": 0.5,
                         "last_price": 2400.0, "in_universe_since": now - 3600, "updated_at": now}])
    s.upsert_candles("ETHUSDT", "5m", [(now // 300 * 300 - 300 * i, 1, 2, 0.5, 1.5, 10) for i in range(5)])
    s.add_signal(now, "ETHUSDT", 0.8, 0.1, "long", [{"feature": "f", "label": "l", "value": 1.0, "contribution": 0.2}])
    s.insert_position({"symbol": "ETHUSDT", "side": "long", "entry_price": 2400.0, "qty": 0.066, "notional": 158.4, "leverage": 8,
                       "margin": 19.8, "sl_price": 2250.0, "tp_price": 2475.0, "init_sl_price": 2250.0, "opened_at": now - 100,
                       "prob": 0.8, "mark_price": 2410.0, "upnl": 0.66, "reasons": []})
    s.insert_trade({"symbol": "BTCUSDT", "side": "short", "entry_price": 1, "exit_price": 0.9, "qty": 1, "opened_at": now - 500,
                    "closed_at": now - 400, "pnl": 5.0, "fees": 0.16, "funding": 0.0, "exit_reason": "tp", "prob": 0.85})
    s.insert_trade({"symbol": "BTCUSDT", "side": "long", "entry_price": 1, "exit_price": 0.9, "qty": 1, "opened_at": now - 300,
                    "closed_at": now - 200, "pnl": -10.0, "fees": 0.16, "funding": 0.0, "exit_reason": "sl", "prob": 0.75})
    s.add_equity(now - 60, 100.0)
    s.add_equity(now, 104.0)
    s.add_log("INFO", "hello")
    s.add_lab_report("backtest", "BT 1", {"sharpe": 1.2}, [{"t": 1, "equity": 100.0}])


def test_health_without_engine(env):
    c, s = env
    h = c.get("/api/health").json()
    assert h == {"ok": True, "engine_alive": False, "engine_heartbeat_age_s": None, "mode": "paper_local", "version": "2.0.0"}


def test_health_with_engine(env):
    c, s = env
    seed(s)
    h = c.get("/api/health").json()
    assert h["engine_alive"] is True and 0 <= h["engine_heartbeat_age_s"] < 5
    s.execute("UPDATE heartbeat SET t=?", (time.time() - 60,))
    h = c.get("/api/health").json()
    assert h["engine_alive"] is False and h["engine_heartbeat_age_s"] > 50


def test_status_shape_and_values(env):
    c, s = env
    seed(s)
    st = c.get("/api/status").json()
    assert set(st) == STATUS_KEYS
    assert st["equity"] == 104.0 and st["peak_equity"] == 110.0 and st["running"] is True
    assert st["dd_basis"] == "start" and st["drawdown_pct"] == 0.0   # 104 is above the 100 stake (peak 110 ignored)
    assert st["open_positions"] == 1 and st["max_positions"] == 5 and st["trades_today"] == 2
    assert st["realized_pnl"] == pytest.approx(5.0 - 10.0)        # sum of NET pnl
    assert st["win_rate"] == 0.5
    assert st["kill"] == {"tripped": False, "reason": "", "dd_limit_pct": 50.0}
    assert set(st["model"]) == {"name", "version", "trained_at", "threshold_long", "threshold_short"}
    assert st["paused_symbols"] == []
    assert set(st["safety"]) == {"state", "reasons", "derate_mult", "drift", "faults"}
    assert st["safety"]["state"] == "NORMAL" and set(st["safety"]["drift"]) == {"z", "n", "mult"}


def test_status_defaults_before_engine_ever_ran(env):
    c, s = env
    st = c.get("/api/status").json()
    assert set(st) == STATUS_KEYS and st["running"] is False and st["equity"] == 100.0 and st["open_positions"] == 0


def test_status_safety_merges_guard_and_engine(env):
    c, s = env
    seed(s)
    s.kv_set("safety_guard", {"state": "DEGRADED", "reasons": ["engine heartbeat stale 40s"], "t": time.time()})
    s.kv_set("safety_engine", {"state": "NORMAL", "reasons": [], "derate_mult": 0.5, "drift": {"z": -2.4, "n": 30, "mult": 0.5},
                               "faults": [{"symbol": "ETHUSDT", "kind": "big_move", "count": 3, "since": 1}]})
    sf = c.get("/api/status").json()["safety"]
    assert sf["state"] == "DEGRADED" and sf["reasons"] == ["engine heartbeat stale 40s"] and sf["derate_mult"] == 0.5
    assert sf["drift"] == {"z": -2.4, "n": 30, "mult": 0.5} and sf["faults"][0]["kind"] == "big_move"


def test_market_universe_candles_signals(env):
    c, s = env
    seed(s)
    m = c.get("/api/market").json()
    assert set(m) == {"btc_price", "btc_change_24h_pct", "btc_trend", "market_volatility_pct", "avg_funding_rate",
                      "long_short_ratio", "updated_at"} and m["btc_trend"] == "down"
    u = c.get("/api/universe").json()
    assert set(u[0]) == {"symbol", "score", "atr_pct", "quote_volume_24h", "change_24h_pct", "funding_rate", "open_interest_usd",
                         "oi_change_1h_pct", "last_price", "in_universe_since", "paused"}
    cd = c.get("/api/candles", params={"symbol": "ethusdt", "tf": "5m", "limit": 3}).json()
    assert len(cd) == 3 and set(cd[0]) == {"t", "o", "h", "l", "c", "v"} and cd[0]["t"] < cd[-1]["t"]
    assert c.get("/api/candles").status_code == 422                       # symbol is required
    sg = c.get("/api/signals", params={"symbol": "ETHUSDT"}).json()
    assert set(sg[0]) == {"t", "symbol", "prob_long", "prob_short", "action", "reasons"}
    assert set(sg[0]["reasons"][0]) == {"feature", "label", "value", "contribution"}
    assert c.get("/api/signals", params={"symbol": "NOPE"}).json() == []


def test_positions_trades_equity_logs_lab(env):
    c, s = env
    seed(s)
    p = c.get("/api/positions").json()
    assert set(p[0]) == {"id", "symbol", "side", "entry_price", "qty", "notional", "leverage", "margin", "sl_price", "tp_price",
                         "trail_active", "trail_price", "mark_price", "upnl", "opened_at", "prob", "reasons"}
    assert p[0]["trail_active"] is False
    t = c.get("/api/trades").json()
    assert set(t[0]) == {"id", "symbol", "side", "entry_price", "exit_price", "qty", "opened_at", "closed_at", "pnl", "fees",
                         "funding", "exit_reason", "prob"}
    assert t[0]["exit_reason"] == "sl" and t[1]["exit_reason"] == "tp"      # newest first
    e = c.get("/api/equity").json()
    assert e[0].keys() == {"t", "equity"} and e[0]["t"] < e[-1]["t"]
    lg = c.get("/api/logs").json()
    assert lg[-1] == {"t": lg[-1]["t"], "level": "INFO", "msg": "hello"}
    lab = c.get("/api/lab/reports").json()
    assert set(lab[0]) == {"id", "created_at", "kind", "title", "summary", "equity"} and lab[0]["kind"] == "backtest"


def test_limits_validated(env):
    c, _ = env
    assert c.get("/api/trades", params={"limit": 0}).status_code == 422
    assert c.get("/api/candles", params={"symbol": "X", "limit": 99999}).status_code == 422


def test_settings_get_put_merge(env):
    c, s = env
    g = c.get("/api/settings").json()
    assert set(g) == {"slice_margin_usd", "leverage", "max_positions", "max_leverage", "max_total_leverage", "max_margin_ratio",
                      "max_open_risk_pct", "exit_mode", "tp_usd", "sl_usd", "tp_atr", "sl_atr", "trail",
                      "short_enabled", "prob_threshold_long", "prob_threshold_short", "corr_cap", "universe_size",
                      "universe_refresh_s", "universe_atr_bars", "universe_retention", "universe_whitelist", "kill_dd_pct", "dd_basis", "fee_taker", "sound", "capital_usd", "decision_every_bars", "cooldown_bars",
                          "max_trades_per_day", "entry_mode", "maker_timeout_s", "maker_reprices", "maker_on_timeout",
                          "breakeven_r", "sl_cooldown_bars", "max_side_risk_pct", "partial_tp_r", "partial_tp_frac"}
    assert g["slice_margin_usd"] == 20 and g["leverage"] == 8 and g["trail"] == {"enabled": True, "activate_r": 1.5, "distance_atr": 1.0}
    r = c.put("/api/settings", json={"leverage": 5, "trail": {"enabled": False}, "short_enabled": True})
    assert r.status_code == 200 and r.json()["leverage"] == 5 and r.json()["trail"]["enabled"] is False
    assert r.json()["trail"]["activate_r"] == 1.5 and r.json()["tp_usd"] == 5          # partial merge
    assert c.get("/api/settings").json() == r.json()
    assert s.get_settings().leverage == 5                                              # engine sees it through the store
    for bad in ({"leverage": 0}, {"exit_mode": "wat"}, {"nonsense": 1}, {"paper_start_equity": 1}, {"prob_threshold_long": 2}):
        assert c.put("/api/settings", json=bad).status_code == 422
    assert c.get("/api/settings").json() == r.json()
    assert g["universe_whitelist"] is None
    w = c.put("/api/settings", json={"universe_whitelist": ["btcusdt", "ETHUSDT"]})
    assert w.status_code == 200 and w.json()["universe_whitelist"] == ["BTCUSDT", "ETHUSDT"]
    assert s.get_settings().universe_whitelist == ["BTCUSDT", "ETHUSDT"]
    for bad in ([], ["not a symbol"], 5):
        assert c.put("/api/settings", json={"universe_whitelist": bad}).status_code == 422
    assert c.put("/api/settings", json={"universe_whitelist": None}).json()["universe_whitelist"] is None


def test_control_validation(env):
    c, s = env
    seed(s)
    assert c.post("/api/control", json={"action": "explode"}).json()["ok"] is False
    assert "position_id" in c.post("/api/control", json={"action": "close_position"}).json()["error"]
    assert "model" in c.post("/api/control", json={"action": "promote_model"}).json()["error"]
    assert s.pending_commands() == []


def test_control_rejected_when_engine_offline(env):
    c, s = env
    r = c.post("/api/control", json={"action": "kill"}).json()
    assert r["ok"] is False and "not running" in r["error"] and s.pending_commands() == []


def test_control_queues_commands_for_engine(env):
    c, s = env
    seed(s)
    for body in ({"action": "start"}, {"action": "close_position", "position_id": 1}, {"action": "kill"},
                 {"action": "promote_model", "model": "m2"}, {"action": "rollback_model", "model": "m1"}):
        assert c.post("/api/control", json=body).json() == {"ok": True, "queued": True}
    cmds = s.pending_commands()
    assert [x["action"] for x in cmds] == ["start", "close_position", "kill", "promote_model", "rollback_model"]
    assert cmds[1]["payload"] == {"position_id": 1} and cmds[3]["payload"] == {"model": "m2"}


def test_control_returns_engine_result(tmp_path):
    db = tmp_path / "tb.sqlite"
    s = Store(db)
    seed(s)
    c = TestClient(create_app(db, control_wait_s=3.0))
    stop = threading.Event()

    def fake_engine():
        s2 = Store(db)
        while not stop.is_set():
            for cmd in s2.pending_commands():
                s2.finish_command(cmd["id"], cmd["action"] != "close_position", None if cmd["action"] != "close_position" else "no such position")
            time.sleep(0.05)

    th = threading.Thread(target=fake_engine, daemon=True)
    th.start()
    try:
        assert c.post("/api/control", json={"action": "stop"}).json() == {"ok": True}
        assert c.post("/api/control", json={"action": "close_position", "position_id": 42}).json() == {"ok": False, "error": "no such position"}
    finally:
        stop.set()
        th.join(2)


def test_shadow_endpoint(env, tmp_path, monkeypatch):
    c, s = env
    monkeypatch.setenv("TB_MODELS", str(tmp_path / "models"))
    d = tmp_path / "models"
    d.mkdir()
    import json
    (d / "champ.json").write_text(json.dumps({"name": "champ", "status": "champion"}))
    (d / "chal.json").write_text(json.dumps({"name": "chal", "status": "challenger"}))
    t0 = int(time.time()) - 1000
    for m, pnls in (("champ", [1.0, -0.5, 0.5]), ("chal", [2.0, 1.0, -0.5, 0.5])):
        for i, p in enumerate(pnls):
            s.add_shadow_trade({"model": m, "symbol": "A", "side": "long", "entry_time": t0 + i, "exit_time": t0 + i + 1,
                                "entry_price": 1, "exit_price": 1, "qty": 1, "pnl": p, "exit_reason": "tp"})
    out = {r["model"]: r for r in c.get("/api/shadow").json()}
    assert set(out["chal"]) == {"model", "status", "kind", "entry_mode", "loaded", "n_trades", "win_rate", "expectancy", "pnl", "vs_champion"}
    assert out["champ"]["status"] == "champion" and out["champ"]["n_trades"] == 3 and out["champ"]["vs_champion"] is None
    assert out["chal"]["status"] == "challenger" and out["chal"]["n_trades"] == 4
    assert out["chal"]["expectancy"] == pytest.approx(0.75) and out["chal"]["vs_champion"] == pytest.approx(0.75 - 1.0 / 3)
    assert out["chal"]["win_rate"] == 0.75


def test_shadow_equity_endpoint(env, tmp_path, monkeypatch):
    c, s = env
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setenv("TB_MODELS", str(d))
    import json
    (d / "champ.json").write_text(json.dumps({"name": "champ", "status": "champion"}))
    (d / "chal.json").write_text(json.dumps({"name": "chal", "status": "challenger", "kind": "gbm"}))
    (d / "idle.json").write_text(json.dumps({"name": "idle", "status": "challenger"}))
    t0 = 1_700_000_000

    def add(m, entry, exit_, pnl):
        s.add_shadow_trade({"model": m, "symbol": "A", "side": "long", "entry_time": entry, "exit_time": exit_,
                            "entry_price": 1, "exit_price": 1, "qty": 1, "pnl": pnl, "exit_reason": "tp"})

    add("champ", t0 + 10, t0 + 100, 1.0)
    add("champ", t0 + 20, t0 + 50, -0.5)      # closes earlier than the first insert: ordering is by exit time
    add("champ", t0 + 30, t0 + 100, 0.25)     # same close second as the first: merged into one point
    add("chal", t0 + 5, t0 + 60, -2.0)
    rows = {r["model"]: r for r in c.get("/api/shadow/equity").json()}
    assert set(rows) == {"champ", "chal", "idle"}
    assert set(rows["champ"]) == {"model", "status", "kind", "points"}
    assert rows["champ"]["status"] == "champion" and rows["chal"]["status"] == "challenger"
    assert rows["chal"]["kind"] == "gbm" and rows["champ"]["kind"] == "per_symbol_5m"
    assert rows["champ"]["points"] == [{"t": t0 + 10, "pnl": 0.0}, {"t": t0 + 50, "pnl": -0.5}, {"t": t0 + 100, "pnl": 0.75}]
    assert rows["chal"]["points"] == [{"t": t0 + 5, "pnl": 0.0}, {"t": t0 + 60, "pnl": -2.0}]
    # a model without trades still gets its single zero point
    assert len(rows["idle"]["points"]) == 1 and rows["idle"]["points"][0]["pnl"] == 0.0
    # the final value equals /api/shadow's total pnl
    tot = {r["model"]: r["pnl"] for r in c.get("/api/shadow").json()}
    for m, r in rows.items():
        assert r["points"][-1]["pnl"] == pytest.approx(tot[m])
    # limit keeps the newest points (cumulative values stay absolute)
    lim = {r["model"]: r for r in c.get("/api/shadow/equity?limit=2").json()}
    assert lim["champ"]["points"] == [{"t": t0 + 50, "pnl": -0.5}, {"t": t0 + 100, "pnl": 0.75}]
    # trades opened before the stake started are ignored (like /api/equity)
    s.kv_set("capital_since_t", t0 + 8)
    rows = {r["model"]: r for r in c.get("/api/shadow/equity").json()}
    assert rows["champ"]["points"][-1]["pnl"] == pytest.approx(0.75)
    assert rows["chal"]["points"] == [{"t": t0 + 8, "pnl": 0.0}]
    assert c.get("/api/shadow/equity?limit=1").status_code == 422


def test_cors_localhost_only(env):
    c, _ = env
    ok = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    ok = c.get("/api/health", headers={"Origin": "http://127.0.0.1:8787"})
    assert ok.headers.get("access-control-allow-origin") == "http://127.0.0.1:8787"
    bad = c.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_static_ui_mounted_and_api_routes_win(env):
    c, _ = env
    assert c.get("/").text == "<html>ui</html>"
    assert c.get("/api/health").json()["ok"] is True


def test_websocket_streams_events(env):
    c, s = env
    seed(s)
    with c.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "status" and set(first["data"]) == STATUS_KEYS
        s.add_event("tick", {"symbol": "ETHUSDT", "t": 1, "price": 2400.0})
        s.add_event("safety", {"state": "DEGRADED", "reasons": ["x"], "derate_mult": 1.0, "drift": {}, "faults": []})
        s.add_event("shadow_trade", {"model": "m", "pnl": 1.0})
        s.add_event("status", {"equity": 1})
        got = {}
        for _ in range(8):
            m = ws.receive_json()
            got.setdefault(m["type"], m["data"])
            if {"tick", "safety", "shadow_trade"} <= set(got):
                break
        assert got["tick"] == {"symbol": "ETHUSDT", "t": 1, "price": 2400.0}
        assert got["safety"]["state"] == "DEGRADED" and got["shadow_trade"]["model"] == "m"


def test_websocket_does_not_replay_old_events(env):
    c, s = env
    s.add_event("tick", {"symbol": "OLD", "t": 1, "price": 1.0})
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "status"
        s.add_event("tick", {"symbol": "NEW", "t": 2, "price": 2.0})
        for _ in range(6):
            m = ws.receive_json()
            if m["type"] == "tick":
                assert m["data"]["symbol"] == "NEW"
                break
        else:
            raise AssertionError("no tick received")


def test_equity_history_starts_at_the_capital_stake(env):
    from tb import capital

    c, s = env
    for t, eq in ((1000, 5000.0), (2000, 4999.0), (3000, 100.0), (4000, 101.5)):
        s.add_equity(t, eq)
    assert len(c.get("/api/equity").json()) == 4               # no stake: full history
    s.kv_set(capital.SINCE_KEY, 3000)
    assert [r["equity"] for r in c.get("/api/equity").json()] == [100.0, 101.5]


def test_capacity_settings_defaults_and_validation(env):
    c, s = env
    g = c.get("/api/settings").json()
    assert g["max_positions"] is None and g["max_leverage"] == 20 and g["max_total_leverage"] == 20
    assert g["max_margin_ratio"] == 0.5 and g["max_open_risk_pct"] == 20
    ok = c.put("/api/settings", json={"max_leverage": 50, "max_total_leverage": 12.5, "max_margin_ratio": 0.3,
                                      "max_open_risk_pct": 15, "max_positions": 3}).json()
    assert (ok["max_leverage"], ok["max_total_leverage"], ok["max_margin_ratio"], ok["max_open_risk_pct"], ok["max_positions"]) == (50, 12.5, 0.3, 15, 3)
    assert c.put("/api/settings", json={"max_positions": None}).json()["max_positions"] is None
    for bad in ({"max_leverage": 0}, {"max_leverage": 126}, {"max_margin_ratio": 0.04}, {"max_margin_ratio": 0.95},
                {"max_total_leverage": 0}, {"max_open_risk_pct": 0}, {"max_positions": 0}):
        assert c.put("/api/settings", json=bad).status_code == 422, bad


def test_access_decision_loopback_tailnet_token_and_lan():
    from tb.api import access_decision
    tok = "x" * 32
    assert access_decision("127.0.0.1", None, tok) == "allow" and access_decision("::1", None, "") == "allow"
    assert access_decision("100.64.159.91", tok, tok) == "allow"              # tailnet + token
    assert access_decision("100.64.159.91", None, tok) == "token"             # tailnet, no token
    assert access_decision("100.64.159.91", "wrong", tok) == "token"
    assert access_decision("100.64.159.91", tok, "") == "deny"                # remote access disabled
    assert access_decision("192.168.1.20", tok, tok) == "deny"                # LAN: never
    assert access_decision("8.8.8.8", tok, tok) == "deny" and access_decision("bogus", tok, tok) == "deny"


def test_tailnet_client_sets_cookie_and_lan_client_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("TB_API_TOKEN", "t" * 32)
    db = tmp_path / "tb.sqlite"
    Store(db)
    app = create_app(db, control_wait_s=0.0, ui=tmp_path)
    tail = TestClient(app, client=("100.80.1.2", 5000))
    assert tail.get("/api/health").status_code == 401
    r = tail.get("/api/health?token=" + "t" * 32, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/api/health" and "tb_token" in r.headers["set-cookie"]
    assert tail.get("/api/health", cookies={"tb_token": "t" * 32}).status_code == 200
    lan = TestClient(app, client=("192.168.1.9", 5000))
    assert lan.get("/api/health?token=" + "t" * 32).status_code == 403
