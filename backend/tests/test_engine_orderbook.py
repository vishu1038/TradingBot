"""Order-book bands (bookDepth definition), 5m depth history, engine depth cycle, pause/unpause/refresh."""
import pytest
from fastapi.testclient import TestClient

from conftest import FakeData
from test_engine_core import mk_engine, open_long, run
from tb import orderbook as ob
from tb.api import create_app
from tb.exchange.base import ExchangeError
from tb.exchange.binance import BinanceExchange


def book(mid=100.0, tick=0.1, levels=1000, qty=1.0):
    bids = [(round(mid - tick / 2 - i * tick, 8), qty) for i in range(levels)]
    asks = [(round(mid + tick / 2 + i * tick, 8), qty) for i in range(levels)]
    return bids, asks


def test_band_definition_matches_bookdepth():
    # mid 100, spread 0.1: bids 99.95, 99.85, ... asks 100.05, 100.15, ...  (qty 2 each, tick 0.1)
    bids, asks = book(100.0, 0.1, 100, 2.0)
    s = ob.compute_snapshot(bids, asks, "X", 1000.0)
    assert s["mid"] == pytest.approx(100.0) and s["spread"] == pytest.approx(0.1)
    by = {b["pct"]: b for b in s["bands"]}
    assert list(by) == [0.2, 1.0, 2.0, 3.0, 4.0, 5.0]
    # +-0.2% of mid = +-0.2: bids >= 99.8 -> 99.95, 99.85 (99.75 < 99.8): 2 levels; asks <= 100.2 -> 100.05, 100.15
    assert by[0.2]["bid_depth"] == pytest.approx(4.0) and by[0.2]["ask_depth"] == pytest.approx(4.0)
    assert by[0.2]["bid_notional"] == pytest.approx(2 * (99.95 + 99.85))
    assert by[0.2]["ask_notional"] == pytest.approx(2 * (100.05 + 100.15))
    assert by[1.0]["bid_depth"] == pytest.approx(2.0 * 10) and by[1.0]["ask_depth"] == pytest.approx(2.0 * 10)
    pcts = list(by)
    assert all(by[a]["bid_depth"] <= by[b]["bid_depth"] for a, b in zip(pcts, pcts[1:]))       # cumulative
    assert [len(s["bids"]), len(s["asks"])] == [20, 20] and s["bids"][0][0] > s["bids"][1][0]


def test_truncated_bands_flagged():
    bids, asks = book(100.0, 0.001, 1000, 1.0)                # book only reaches ~1% from mid
    s = ob.compute_snapshot(bids, asks, "X")
    assert s["cover_pct"] == pytest.approx(1.0, abs=0.01)
    by = {b["pct"]: b["truncated"] for b in s["bands"]}
    assert by[0.2] is False and by[1.0] is True and by[2.0] is True and by[5.0] is True
    full = ob.compute_snapshot(*book(100.0, 0.1, 1000), "X")      # reaches 50%
    assert not any(b["truncated"] for b in full["bands"])


def test_empty_or_crossed_book_rejected():
    assert ob.compute_snapshot([], [(1, 1)]) is None
    assert ob.compute_snapshot([(2, 1)], [(1, 1)]) is None


def test_archive_row_shape():
    s = ob.compute_snapshot(*book(100.0, 0.1, 100, 1.0), "X")
    row = ob.snapshot_to_row(s)
    assert set(row) == {c for c in ob.frame_columns() if c != "open_time"}
    assert row["depth_-0.2"] == s["bands"][0]["bid_depth"] and row["notional_5"] == s["bands"][-1]["ask_notional"]
    assert list(ob.PCTS) == [-5, -4, -3, -2, -1, -0.2, 0.2, 1, 2, 3, 4, 5]


def test_depth_frame_keeps_last_snapshot_per_5m_bar(store):
    t0 = 1_800_000_000 // 300 * 300                 # a bar boundary (seconds)
    for off, qty in ((10, 1.0), (100, 2.0), (290, 3.0), (310, 4.0)):    # three in bar 0, one in bar 1
        ob.record_snapshot(store, ob.compute_snapshot(*book(100.0, 0.1, 100, qty), "X", t0 + off))
    # an out-of-order older snapshot must not overwrite a newer one in the same bar
    ob.record_snapshot(store, ob.compute_snapshot(*book(100.0, 0.1, 100, 9.0), "X", t0 + 20))
    df = ob.depth_frame("X", store)
    assert list(df.columns) == ob.frame_columns()
    assert df["open_time"].tolist() == [t0 * 1000, (t0 + 300) * 1000]
    assert df["depth_-1"].tolist() == [30.0, 40.0]            # qty 3 / qty 4 * 10 levels within 1%
    assert df["depth_-1"].dtype == "float64" and df["open_time"].dtype == "int64"
    empty = ob.depth_frame("NOPE", store)
    assert empty.empty and list(empty.columns) == ob.frame_columns()


def test_depth_history_pruned(store):
    t0 = 1_800_000_000 // 300 * 300
    for i in range(ob.KEEP_BARS + 10):
        ob.record_snapshot(store, ob.compute_snapshot(*book(100.0, 0.1, 30, 1.0), "X", t0 + i * 300))
    assert len(ob.depth_frame("X", store)) <= ob.KEEP_BARS + 1


# ------------------------------------------------------------------ binance depth request
def test_binance_depth_request_and_parse():
    class R:
        status_code = 200
        headers = {"X-MBX-USED-WEIGHT-1M": "40"}

        def json(self):
            return {"bids": [["99.9", "2"], ["99.8", "1"]], "asks": [["100.1", "3"]]}

    class S:
        def request(self, method, url, headers=None, params=None, timeout=None):
            self.seen = (method, url, params)
            return R()

    s = S()
    ex = BinanceExchange("https://fapi.binance.com", session=s)
    bids, asks = ex.depth("ETHUSDT", 1000)
    assert s.seen == ("GET", "https://fapi.binance.com/fapi/v1/depth", {"symbol": "ETHUSDT", "limit": "1000"})
    assert bids == [(99.9, 2.0), (99.8, 1.0)] and asks == [(100.1, 3.0)] and ex.used_weight == 40


# ------------------------------------------------------------------ engine depth cycle
class DepthData(FakeData):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []
        self.depth_fail = False
        self.used_weight = 0

    def depth(self, symbol, limit=1000):
        self.calls.append(symbol)
        if self.depth_fail:
            raise ExchangeError("depth down")
        return book(self.marks[symbol], 0.01, 1000, 1.0)


def mk_depth_engine(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    d = DepthData(("AAAUSDT", "BBBUSDT", "BTCUSDT"))
    d.kl, d.marks = data.kl, data.marks
    eng.data = d
    eng.ex.data = d
    return eng, d


def test_depth_cycle_round_robin_store_event_and_history(store, tmp_path):
    eng, d = mk_depth_engine(store, tmp_path)
    run(eng.depth_cycle())
    run(eng.depth_cycle())
    assert sorted(d.calls) == ["AAAUSDT", "BBBUSDT"]            # one symbol per call, oldest first
    run(eng.depth_cycle())
    assert len(d.calls) == 2                                    # both fresher than the min age: no extra request
    snap = store.get_orderbook("AAAUSDT")
    assert snap["mid"] == pytest.approx(d.marks["AAAUSDT"], rel=1e-3) and len(snap["bands"]) == 6
    ev = [e for e in store.events_after(0) if e["type"] == "orderbook"]
    assert {e["data"]["symbol"] for e in ev} == {"AAAUSDT", "BBBUSDT"}
    assert not ob.depth_frame("AAAUSDT", store).empty
    base = eng.clock()
    eng.clock = lambda: base + 100
    run(eng.depth_cycle())
    assert len(d.calls) == 3                                    # aged out -> refreshed again


def test_depth_failures_and_weight_budget_never_affect_trading(store, tmp_path):
    eng, d = mk_depth_engine(store, tmp_path)
    d.depth_fail = True
    run(eng.depth_cycle())                                      # swallowed
    assert eng.depth_stats["err"] == 1 and store.get_orderbook("AAAUSDT") is None
    assert not eng.faults.exchange_fault()
    d.depth_fail = False
    d.used_weight = 2000                                        # over the budget: skip, keep headroom for trading calls
    eng.depth_at.clear()
    run(eng.depth_cycle())
    assert d.calls == ["AAAUSDT"]                               # only the failed attempt
    open_long(eng, "AAAUSDT")
    assert len(store.get_positions()) == 1


# ------------------------------------------------------------------ pause / unpause / refresh
def test_pause_blocks_new_entries_only_and_persists(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng, "AAAUSDT")
    assert run(eng.handle_command("pause_symbol", {"symbol": "aaausdt"})) is None
    assert "AAAUSDT" in eng.paused and store.kv_get("paused_symbols") == ["AAAUSDT"]
    assert len(store.get_positions()) == 1                       # open trade untouched
    store.execute("DELETE FROM positions")
    eng.ex.pos.clear()
    open_long(eng, "AAAUSDT")
    assert store.get_positions() == []                           # no new entry
    open_long(eng, "BBBUSDT")
    assert len(store.get_positions()) == 1                       # other coins unaffected
    eng2, data2 = mk_engine(store, tmp_path)
    assert eng2.paused == {"AAAUSDT"}                            # persisted across a restart
    assert run(eng2.handle_command("unpause_symbol", {"symbol": "AAAUSDT"})) is None
    assert eng2.paused == set() and store.kv_get("paused_symbols") == []
    assert eng2.entry_allowed("AAAUSDT")
    assert "symbol is required" in run(eng2.handle_command("pause_symbol", {}))
    assert "symbol is required" in run(eng2.handle_command("pause_symbol", {"symbol": "A;B"}))


def test_paused_position_still_exits_at_tp(store, tmp_path):
    eng, data = mk_engine(store, tmp_path)
    open_long(eng, "AAAUSDT")
    run(eng.handle_command("pause_symbol", {"symbol": "AAAUSDT"}))
    data.marks["AAAUSDT"] = store.get_positions()[0]["tp_price"] * 1.01
    run(eng.prices_cycle())
    assert store.get_trades(1)[0]["exit_reason"] == "tp"


def test_refresh_universe_forces_rerank_now(store, tmp_path, monkeypatch):
    import tb.universe as uni

    eng, data = mk_engine(store, tmp_path)
    eng.universe_at = eng.clock()                                # fresh: a normal cycle would do nothing
    calls = []
    monkeypatch.setattr(uni, "refresh", lambda *a, **k: calls.append(1) or [
        {"symbol": "BBBUSDT", "rank": 1, "score": 1.0, "last_price": 1.0, "in_universe_since": 1, "updated_at": 1}])
    run(eng.universe_cycle())
    assert calls == [] and eng.universe == ["AAAUSDT", "BBBUSDT"]
    assert run(eng.handle_command("refresh_universe", {})) is None     # returns at once, the loop does the work
    assert eng.force_universe is True
    run(eng.universe_cycle())
    assert calls == [1] and eng.universe == ["BBBUSDT"] and eng.force_universe is False
    run(eng.universe_cycle())
    assert calls == [1]                                          # flag consumed


# ------------------------------------------------------------------ API
@pytest.fixture
def api(tmp_path):
    from tb.store import Store

    db = tmp_path / "tb.sqlite"
    s = Store(db)
    s.beat("paper_local", "2.0.0", 1)
    return TestClient(create_app(db, control_wait_s=0.0, ui=tmp_path / "noui")), s


def test_api_orderbook_shape_and_404(api):
    c, s = api
    s.set_orderbook("ETHUSDT", ob.compute_snapshot(*book(2400.0, 0.1, 1000, 5.0), "ETHUSDT", 1_800_000_000))
    r = c.get("/api/orderbook", params={"symbol": "ethusdt"})
    assert r.status_code == 200
    j = r.json()
    assert set(j) == {"symbol", "t", "mid", "spread", "spread_pct", "cover_pct", "bands", "bids", "asks"}
    assert j["symbol"] == "ETHUSDT" and j["t"] == 1_800_000_000 and j["mid"] == pytest.approx(2400.0)
    assert set(j["bands"][0]) == {"pct", "bid_depth", "bid_notional", "ask_depth", "ask_notional", "truncated"}
    assert [b["pct"] for b in j["bands"]] == [0.2, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert len(j["bids"]) == 20 and len(j["asks"]) == 20 and len(j["bids"][0]) == 2
    assert c.get("/api/orderbook", params={"symbol": "NOPEUSDT"}).status_code == 404
    assert c.get("/api/orderbook").status_code == 422


def test_api_control_and_universe_paused_flag(api):
    c, s = api
    s.replace_universe([{"symbol": "ETHUSDT", "rank": 1, "score": 1.0}, {"symbol": "BTCUSDT", "rank": 2, "score": 1.0}])
    assert c.post("/api/control", json={"action": "pause_symbol"}).json()["ok"] is False
    assert c.post("/api/control", json={"action": "pause_symbol", "symbol": "bad sym"}).json()["ok"] is False
    assert c.post("/api/control", json={"action": "pause_symbol", "symbol": "ethusdt"}).json() == {"ok": True, "queued": True}
    assert c.post("/api/control", json={"action": "unpause_symbol", "symbol": "ETHUSDT"}).json()["ok"] is True
    assert c.post("/api/control", json={"action": "refresh_universe"}).json()["ok"] is True
    cmds = s.pending_commands()
    assert [(x["action"], x["payload"]) for x in cmds] == [("pause_symbol", {"symbol": "ETHUSDT"}),
                                                           ("unpause_symbol", {"symbol": "ETHUSDT"}),
                                                           ("refresh_universe", {})]
    s.kv_set("paused_symbols", ["ETHUSDT"])
    u = {r["symbol"]: r["paused"] for r in c.get("/api/universe").json()}
    assert u == {"ETHUSDT": True, "BTCUSDT": False}


def test_paper_test_only_model_refused_outside_paper(store, tmp_path, monkeypatch):
    import json

    import tb.engine as E
    from tb import scorer as S
    from test_engine_core import FixedScorer

    d = tmp_path / "models"
    d.mkdir()
    (d / "ptest.json").write_text(json.dumps({"name": "ptest", "status": "champion", "created_at": "2026-10-01T00:00:00Z",
                                              "note": "PAPER TEST ONLY - no out-of-sample edge. Never use for demo/live."}))
    (d / "ok.json").write_text(json.dumps({"name": "ok", "status": "challenger"}))
    monkeypatch.setattr(E, "load_model", lambda e: (FixedScorer(0.9), S.ModelInfo(e.name, "1", None, 0.7, 0.8, e.status)))
    paper, _ = mk_engine(store, tmp_path)
    assert paper.model_info.name == "ptest" and not isinstance(paper.scorer, S.NullScorer)
    for mode in ("binance_demo", "live"):
        eng, _ = mk_engine(store, tmp_path, mode=mode)
        assert isinstance(eng.scorer, S.NullScorer) and eng.model_info is None, mode
        assert set(eng.challengers) == {"ok"}                        # unmarked models still load
    # a paper-only challenger is not loaded either, and cannot be promoted
    (d / "ptest.json").write_text(json.dumps({"name": "ptest", "status": "challenger", "note": "paper test only"}))
    (d / "ok.json").write_text(json.dumps({"name": "ok", "status": "champion"}))
    eng, _ = mk_engine(store, tmp_path, mode="binance_demo")
    assert eng.model_info.name == "ok" and "ptest" not in eng.challengers
    assert "PAPER TEST ONLY" in run(eng.handle_command("promote_model", {"model": "ptest"}))


def test_ws_forwards_orderbook_events(api):
    c, s = api
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "status"
        snap = ob.compute_snapshot(*book(100.0, 0.1, 100, 1.0), "AAAUSDT", 1_800_000_000)
        s.add_event("orderbook", snap)
        for _ in range(6):
            m = ws.receive_json()
            if m["type"] == "orderbook":
                assert m["data"]["symbol"] == "AAAUSDT" and len(m["data"]["bands"]) == 6
                break
        else:
            raise AssertionError("no orderbook message")
