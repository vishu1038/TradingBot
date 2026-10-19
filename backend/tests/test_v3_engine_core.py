"""V3 engine: scorer interface v2 / loader dispatch, history paging, entry controls (cadence, cooldown, max trades per
day), model-driven exits (pct, atr_tf), shadow challengers of any kind (+ maker entries), xsec rebalances, status."""
import json
import time

import numpy as np
import pandas as pd
import pytest

from conftest import FakeData, make_klines
from test_engine_core import BAR, FAST, closed_of, load_bars, mk_engine, run, write_model
from tb import engine as E
from tb import scorer as S
from tb.config import ROOT, Settings
from tb.entry import EntryCfg, EntryGate, XsecCfg, is_decision_bar, next_decision_ms
from tb.exchange.base import SymbolFilters
from tb.exits import ExitSpec, bind, exit_atr, resample_closed, resolve_exit
from tb.shadow import ShadowBook

HOUR = 3600


def hour_now():
    return int(time.time() // HOUR * HOUR) + 10.0          # 10 s after a full hour: the newest closed bar closes ON the hour


class V2:
    """A Scorer-interface-v2 model double."""
    version = "1"

    def __init__(self, name, kind="mtf", pl=0.9, ps=0.0, hist=100):
        self.name, self.kind, self.history_bars_5m, self.pl, self.ps = name, kind, hist, pl, ps
        self.calls = []

    def score(self, symbol, feats):
        return S.Score(self.pl, self.ps, [S.Reason("f", "l", 1.0, 0.1)])

    def score_universe(self, frames):
        self.calls.append(sorted(frames))
        return {s: self.score(s, None) for s in frames}


class Ranker(V2):
    """xsec double: fixed per-symbol percentile."""

    def __init__(self, name, pct):
        super().__init__(name, "xsec", hist=100)
        self.pct = pct

    def score_universe(self, frames):
        self.calls.append(sorted(frames))
        return {s: S.Score(self.pct.get(s, 0.0), (1 - self.pct[s]) if s in self.pct else 0.0, []) for s in frames}


def info_for(entry):
    m = entry.meta
    return S.ModelInfo(entry.name, "1", entry.created_at, m.get("thresholds", {}).get("long"),
                       m.get("thresholds", {}).get("short"), entry.status, m.get("kind", "per_symbol_5m"),
                       m.get("history_bars_5m", 0), tuple(m.get("timeframes", ["5m"])), m.get("entry_cfg", {}),
                       m.get("exit_cfg", {}), m)


def write_v3(d, name, status, **meta):
    d.mkdir(exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps({"name": name, "version": "1", "created_at": "2026-10-01T00:00:00Z",
                                                "status": status, "thresholds": {"long": 0.7, "short": 0.8}, **meta}))


@pytest.fixture
def doubles(monkeypatch):
    made = {}

    def fake_load(entry):
        sc = made.get(entry.name) or V2(entry.name, entry.meta.get("kind", "mtf"), hist=entry.meta.get("history_bars_5m", 100))
        made[entry.name] = sc
        return sc, info_for(entry)

    monkeypatch.setattr(E, "load_model", fake_load)
    return made


# ------------------------------------------------------------------ scorer loader
def test_unknown_and_broken_kinds_are_logged_and_skipped(tmp_path, caplog):
    d = tmp_path / "models"
    write_v3(d, "quantum", "challenger", kind="quantum")
    write_v3(d, "broken_mtf", "challenger", kind="mtf")              # no weights next to the json
    write_v3(d, "broken_gbm", "challenger", kind="gbm")
    write_v3(d, "broken_xsec", "challenger", kind="xsec")
    for e in S.scan_models(d):
        assert S.load_model(e) is None
    assert "unknown kind" in caplog.text


def test_engine_survives_models_that_cannot_load(store, tmp_path):
    d = tmp_path / "models"
    write_v3(d, "quantum", "challenger", kind="quantum")
    write_v3(d, "broken", "champion", kind="gbm")
    eng, _ = mk_engine(store, tmp_path)
    assert isinstance(eng.scorer, S.NullScorer) and eng.challengers == {}
    run(eng.bars_cycle())                                            # still runs


def test_legacy_artifact_loads_as_per_symbol_5m_with_default_history():
    # legacy = an artifact without a V3 `kind` (the models/ dir also holds V3 exports, newest first)
    ent = [e for e in S.scan_models(ROOT / "models") if e.meta.get("arch") and not e.meta.get("kind")] \
        if (ROOT / "models").is_dir() else []
    if not ent:
        pytest.skip("no exported model in models/")
    got = S.load_model(ent[0])
    assert got is not None
    sc, info = got
    from tb import features
    assert info.kind == "per_symbol_5m" and info.history_bars_5m == features.WARMUP + features.WINDOW
    assert info.entry_cfg == {} or isinstance(info.entry_cfg, dict)


def test_score_frames_prefers_score_universe_and_falls_back_per_symbol():
    b = S.FrameBundle("A", pd.DataFrame())
    assert b.klines is b.kl and b.btc is b.btc_kl is None               # aliases the other modules read
    sc = V2("m")
    assert S.score_frames(sc, None, {"A": b})["A"].prob_long == 0.9 and sc.calls == [["A"]]

    class OnlyScore:
        def score(self, sym, feats):
            return S.Score(0.5, 0.1, [])

    kl = make_klines(2500, end_ms=int(time.time() * 1000))
    out = S.score_frames(OnlyScore(), S.ModelInfo("o", "1", None, None, None), {"A": S.FrameBundle("A", kl, None, None, None, kl)})
    assert out["A"].prob_long == 0.5                                     # classic per-symbol window path


# ------------------------------------------------------------------ entry controls
def test_decision_grid_is_utc_aligned():
    h = 1_700_000_000_000 // 3_600_000 * 3_600_000                     # a full hour (ms)
    assert is_decision_bar(h - 300_000, 12) and not is_decision_bar(h, 12) and not is_decision_bar(h - 600_000, 12)
    assert is_decision_bar(h - 300_000, 1) and is_decision_bar(h, 1)
    assert next_decision_ms(h + 1, 12) == h + 3_600_000 and next_decision_ms(h - 1, 12) == h


def test_entry_cfg_overlays_the_settings_defaults_per_key():
    cfg = Settings().merge({"cooldown_bars": 6, "max_trades_per_day": 5}, allow_internal=True)
    e = EntryCfg.resolve(cfg, {"decision_every_bars": 12, "entry_mode": "maker", "max_trades_per_day": None, "bogus": 1})
    assert (e.decision_every_bars, e.cooldown_bars, e.max_trades_per_day, e.entry_mode) == (12, 6, None, "maker")
    assert EntryCfg.resolve(cfg, {"cooldown_bars": "x", "entry_mode": "weird"}).cooldown_bars == 6
    x = XsecCfg.resolve({"top_k": 9}, {"rebalance_every_h": 4, "xsec_cfg": {"bottom_k": 2}})
    assert (x.rebalance_every_h, x.top_k, x.bottom_k) == (4, 9, 2)


def test_gate_cooldown_counts_from_the_later_of_entry_and_exit_and_max_per_day():
    cfg = EntryCfg(cooldown_bars=12, max_trades_per_day=2)
    g = EntryGate()
    t = 1_700_000_000.0
    g.record_entry("A", t)
    g.record_exit("A", t + 1800)
    assert not g.check("A", t + 1800 + 3000, cfg)[0]                    # < 1 h after the EXIT
    assert g.check("A", t + 1800 + 3700, cfg)[0] and g.check("B", t + 10, cfg)[0]
    g.record_entry("B", t + 20)
    ok, why = g.check("C", t + 30, cfg)
    assert not ok and "max_trades_per_day" in why
    assert g.check("C", t + 90_000, cfg)[0]                             # next UTC day


def test_champion_decides_only_on_its_cadence_and_respects_max_trades_per_day(store, tmp_path, doubles):
    d = tmp_path / "models"
    write_v3(d, "hourly", "champion", kind="mtf", history_bars_5m=100,
             entry_cfg={"decision_every_bars": 12, "max_trades_per_day": 1}, thresholds={"long": 0.5, "short": 0.9})
    eng, _ = mk_engine(store, tmp_path, now=hour_now())
    assert eng.model_info.kind == "mtf"
    run(eng.bars_cycle())
    assert len(store.get_positions()) == 1                               # two coins signal, the cap lets one trade
    assert len(store.get_signals(None, 10)) == 2
    st = eng.decisions_status()
    assert st["trades_today"] == 1 and st["max_trades_per_day"] == 1 and st["decision_every_bars"] == 12
    # 5 minutes later the newest bar is not an hourly decision point: nothing is scored or recorded
    eng2, _ = mk_engine(store, tmp_path, now=hour_now() + 300)
    n = len(store.get_signals(None, 100))
    run(eng2.bars_cycle())
    assert len(store.get_signals(None, 100)) == n


def test_per_coin_cooldown_blocks_a_reentry(store, tmp_path, doubles):
    d = tmp_path / "models"
    write_v3(d, "cool", "champion", kind="mtf", history_bars_5m=100, entry_cfg={"cooldown_bars": 12})
    eng, _ = mk_engine(store, tmp_path)
    run(eng.bars_cycle())
    pos = store.get_positions()
    assert {p["symbol"] for p in pos} == {"AAAUSDT", "BBBUSDT"}
    run(eng.close_all("manual"))
    assert store.get_positions() == []
    eng.last_closed.clear()
    run(eng.bars_cycle())                                                # same bar again, but the coins are cooling down
    assert store.get_positions() == []
    eng.live_gate.last.clear()                                           # cooldown over
    eng.last_closed.clear()
    run(eng.bars_cycle())
    assert len(store.get_positions()) == 2


def test_live_gate_is_restart_safe(store, tmp_path):
    eng, _ = mk_engine(store, tmp_path)
    now = int(eng.clock())
    store.insert_trade({"symbol": "AAAUSDT", "side": "long", "entry_price": 1, "exit_price": 1, "qty": 1, "opened_at": now - 600,
                        "closed_at": now - 300, "pnl": 0, "exit_reason": "tp"})
    eng.seed_live_gate()
    assert eng.live_gate.trades_today(eng.clock()) == 1
    assert not eng.live_gate.check("AAAUSDT", eng.clock(), EntryCfg(cooldown_bars=6))[0]


# ------------------------------------------------------------------ exits
def test_exit_spec_resolution():
    assert resolve_exit({"exit_mode": "atr", "tp_atr": 3, "sl_atr": 2}).mode == "settings"      # classic: Settings drive
    assert resolve_exit(None).mode == "settings" and resolve_exit({"exit_mode": "fixed"}).mode == "settings"
    p = resolve_exit({"exit_mode": "pct", "tp_pct": 3.0, "sl_pct": 6.0})
    assert (p.mode, p.tp_pct, p.sl_pct, p.tp_enabled) == ("pct", 0.03, 0.06, True)
    a = resolve_exit({"exit_mode": "atr", "atr_tf": "4h", "tp_atr": 2, "sl_atr": 1})
    assert (a.mode, a.atr_tf, a.history_bars()) == ("atr", "4h", 17 * 48)
    assert resolve_exit({"exit_mode": "pct", "tp_pct": 3.0}).mode == "settings"                 # invalid -> safe fallback
    assert resolve_exit({"atr_tf": "3w"}).mode == "settings"
    v = resolve_exit({"exit_mode": "pct", "stop_vol_mult": 2.0, "stop_min": 0.02, "stop_max": 0.2}, "xsec")
    assert v.vol_mult == 2.0 and not v.tp_enabled and v.history_bars() > 2000
    assert p.effective(Settings()).sl_atr == 0.06 and p.atr_arg(1.0, 100.0) == 100.0


def test_resample_uses_only_complete_utc_aligned_bars_and_atr_matches_hourly_bars():
    now_ms = int(hour_now() * 1000)
    kl = make_klines(1000, end_ms=now_ms)
    closed = kl[kl["open_time"] + BAR <= now_ms]
    h = resample_closed(closed, "1h", now_ms)
    assert (h["open_time"] % 3_600_000 == 0).all() and h["open_time"].iloc[-1] + 3_600_000 <= now_ms
    last = closed[closed["open_time"] >= h["open_time"].iloc[-1]].head(12)
    assert h["high"].iloc[-1] == last["high"].max() and h["low"].iloc[-1] == last["low"].min()
    # mid-hour: the forming hour is not used
    part = closed.iloc[:-5]
    hp = resample_closed(part, "1h", int(part["open_time"].iloc[-1]) + BAR)
    assert hp["open_time"].iloc[-1] + 3_600_000 <= int(part["open_time"].iloc[-1]) + BAR
    a5 = exit_atr(ExitSpec(), closed, now_ms)
    a1h = exit_atr(ExitSpec("atr", "1h", tp_atr=2.0, sl_atr=1.0), closed, now_ms)
    assert a1h > a5 > 0


def test_vol_scaled_stop_is_clipped_and_needs_history():
    now_ms = int(hour_now() * 1000)
    closed = make_klines(2400, end_ms=now_ms)
    closed = closed[closed["open_time"] + BAR <= now_ms]
    s = bind(resolve_exit({"exit_mode": "pct", "stop_vol_mult": 2.0, "stop_min": 0.02, "stop_max": 0.2}, "xsec"), closed, now_ms)
    assert 0.02 <= s.sl_pct <= 0.2 and s.vol_mult is None
    assert bind(resolve_exit({"exit_mode": "pct", "stop_vol_mult": 2.0}, "xsec"), closed.tail(300), now_ms) is None


def test_pct_exit_model_drives_stop_target_and_sizing(store, tmp_path, doubles):
    d = tmp_path / "models"
    write_v3(d, "pctm", "champion", kind="mtf", history_bars_5m=100, exit_cfg={"exit_mode": "pct", "tp_pct": 3.0, "sl_pct": 6.0})
    eng, _ = mk_engine(store, tmp_path)
    run(eng.bars_cycle())
    p = store.get_positions()[0]
    assert p["exit_mode"] == "pct"
    assert (p["entry_price"] - p["sl_price"]) / p["entry_price"] == pytest.approx(0.06, abs=1e-3)
    assert (p["tp_price"] - p["entry_price"]) / p["entry_price"] == pytest.approx(0.03, abs=1e-3)
    assert p["qty"] * p["entry_price"] <= 160.5 and (p["entry_price"] - p["sl_price"]) * p["qty"] <= 10.0 + 0.2
    orders = {o.kind: o for o in eng.ex.open_conditional_orders(p["symbol"])}
    assert set(orders) == {"SL", "TP"}


def test_atr_tf_exit_uses_resampled_bars(store, tmp_path, doubles):
    d = tmp_path / "models"
    write_v3(d, "atr1h", "champion", kind="mtf", history_bars_5m=100,
             exit_cfg={"exit_mode": "atr", "atr_tf": "1h", "tp_atr": 2.0, "sl_atr": 1.0})
    eng, _ = mk_engine(store, tmp_path, now=hour_now())
    assert eng.hist_need >= 17 * 12
    run(eng.bars_cycle())
    p = store.get_positions()[0]
    a = exit_atr(ExitSpec("atr", "1h", tp_atr=2.0, sl_atr=1.0), closed_of(eng, p["symbol"]), eng.now_ms())
    assert p["exit_mode"] == "atr:1h"
    assert p["entry_price"] - p["sl_price"] == pytest.approx(a, rel=0.02) and p["tp_price"] - p["entry_price"] == pytest.approx(2 * a, rel=0.02)


def test_classic_models_keep_the_settings_exits(store, tmp_path, doubles):
    d = tmp_path / "models"
    write_v3(d, "classic", "champion", exit_cfg={"exit_mode": "atr", "tp_atr": 3.0, "sl_atr": 2.0})
    eng, _ = mk_engine(store, tmp_path)
    run(eng.bars_cycle())
    p = store.get_positions()[0]
    assert p["exit_mode"] == "fixed" and (p["tp_price"] - p["entry_price"]) * p["qty"] == pytest.approx(5.0, abs=0.1)


# ------------------------------------------------------------------ history paging
class DeepData(FakeData):
    used_weight = 0

    def klines(self, symbol, interval="5m", limit=500, start_ms=None, end_ms=None):
        k = self.kl[symbol]
        if end_ms is not None:
            k = k[k["open_time"] <= end_ms]
        return k.tail(limit).reset_index(drop=True)


def deep_engine(store, tmp_path, need):
    d = tmp_path / "models"
    write_v3(d, "deep", "challenger", kind="mtf", history_bars_5m=need)
    eng, data = mk_engine(store, tmp_path)
    data.__class__ = DeepData
    now_ms = int(eng.clock() * 1000)
    for i, s in enumerate(data.kl):
        data.kl[s] = make_klines(need + 400, seed=i + 1, end_ms=now_ms)
    return eng, data


def test_history_is_paged_back_until_the_largest_model_need_is_met(store, tmp_path, doubles):
    eng, data = deep_engine(store, tmp_path, 3200)
    assert eng.hist_need == 3200
    for _ in range(5):
        run(eng.update_symbol("AAAUSDT"))
    k = eng.kl["AAAUSDT"]
    assert len(k) == 3200 + 40 and k["open_time"].is_monotonic_increasing and k["open_time"].is_unique
    assert (np.diff(k["open_time"].to_numpy()) == BAR).all()
    n = len(k)
    run(eng.update_symbol("AAAUSDT"))
    assert len(eng.kl["AAAUSDT"]) == n                                   # complete: no more history requests


def test_history_backfill_respects_the_request_weight_budget(store, tmp_path, doubles):
    eng, data = deep_engine(store, tmp_path, 3200)
    run(eng.update_symbol("AAAUSDT"))
    first = len(eng.kl["AAAUSDT"])
    data.used_weight = 2000
    run(eng.update_symbol("AAAUSDT"))
    assert len(eng.kl["AAAUSDT"]) == first
    data.used_weight = 0
    eng._backfill_budget = 0                                              # per-cycle page budget used up
    run(eng.update_symbol("AAAUSDT"))
    assert len(eng.kl["AAAUSDT"]) == first
    eng._backfill_budget = 1
    run(eng.update_symbol("AAAUSDT"))
    assert len(eng.kl["AAAUSDT"]) > first


def test_young_listing_marks_history_exhausted_and_scoring_waits(store, tmp_path, doubles):
    eng, data = deep_engine(store, tmp_path, 3200)
    data.kl["AAAUSDT"] = make_klines(900, seed=1, end_ms=int(eng.clock() * 1000))      # only 900 bars exist
    for _ in range(3):
        run(eng.update_symbol("AAAUSDT"))
    assert "AAAUSDT" in eng.hist_done and len(eng.kl["AAAUSDT"]) == 900
    sc = V2("deep")
    r = run(eng.score_one(sc, eng.challengers["deep"][1], "AAAUSDT", closed_of(eng, "AAAUSDT"), {}))
    assert r is None and sc.calls == []                                   # not enough bars for the model: not scored


def test_derivative_lookback_follows_the_model_need(store, tmp_path, doubles):
    eng, _ = deep_engine(store, tmp_path, 3200)
    assert eng.deriv_lookback_h() == pytest.approx(3200 * 5 / 60 + 2)
    eng.hist_need = 9600
    assert eng.deriv_lookback_h() == 28 * 24.0                          # capped by what REST serves
    eng2, _ = mk_engine(store, tmp_path / "other")
    assert eng2.deriv_lookback_h() == 26.0


# ------------------------------------------------------------------ shadow challengers
def test_challengers_of_every_kind_shadow_with_their_own_cadence_and_templates(store, tmp_path, doubles):
    d = tmp_path / "models"
    write_v3(d, "champ", "champion", kind="mtf", history_bars_5m=100)
    write_v3(d, "gbm_h", "challenger", kind="gbm", history_bars_5m=100, entry_cfg={"decision_every_bars": 12},
             exit_cfg={"exit_mode": "pct", "tp_pct": 3.0, "sl_pct": 6.0})
    write_v3(d, "mtf_5", "challenger", kind="mtf", history_bars_5m=100)
    eng, _ = mk_engine(store, tmp_path, now=hour_now() + 300)             # not an hourly decision point
    run(eng.bars_cycle())
    assert eng.shadow.has("mtf_5", "AAAUSDT") and eng.shadow.has("champ", "AAAUSDT")
    assert not eng.shadow.has("gbm_h", "AAAUSDT")                          # waits for its hourly bar
    eng2, _ = mk_engine(store, tmp_path, now=hour_now())
    run(eng2.bars_cycle())
    p = eng2.shadow.pos["gbm_h|AAAUSDT"]
    assert (p.entry - p.sl) / p.entry == pytest.approx(0.06, abs=1e-3) and (p.tp - p.entry) / p.entry == pytest.approx(0.03, abs=1e-3)
    q = eng2.shadow.pos["mtf_5|AAAUSDT"]                                   # Settings exits: fixed $10 / $5 on the notional
    assert (q.entry - q.sl) * q.qty == pytest.approx(10.0, abs=0.2)


def test_shadow_gate_applies_per_model_cooldown_and_daily_cap(store, tmp_path, doubles):
    d = tmp_path / "models"
    write_v3(d, "champ", "champion", kind="mtf", history_bars_5m=100)
    write_v3(d, "capped", "challenger", kind="mtf", history_bars_5m=100, entry_cfg={"max_trades_per_day": 1})
    eng, _ = mk_engine(store, tmp_path)
    run(eng.bars_cycle())
    assert sum(1 for p in eng.shadow.pos.values() if p.model == "capped") == 1
    assert sum(1 for p in eng.shadow.pos.values() if p.model == "champ") == 2


def test_a_failing_model_does_not_stop_the_others(store, tmp_path, doubles):
    d = tmp_path / "models"
    write_v3(d, "champ", "champion", kind="mtf", history_bars_5m=100)
    write_v3(d, "bad", "challenger", kind="mtf", history_bars_5m=100)
    write_v3(d, "good", "challenger", kind="mtf", history_bars_5m=100)
    eng, _ = mk_engine(store, tmp_path)

    def boom(frames):
        raise RuntimeError("model blew up")

    eng.challengers["bad"][0].score_universe = boom
    run(eng.bars_cycle())
    assert eng.shadow.has("good", "AAAUSDT") and not eng.shadow.has("bad", "AAAUSDT")
    assert len(store.get_positions()) == 2


def bars(n=60, seed=3):
    return make_klines(n, seed=seed, end_ms=int(hour_now() * 1000))


def test_shadow_maker_entry_fills_only_when_a_later_bar_trades_through_the_limit():
    cfg = Settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0}, allow_internal=True)
    f = SymbolFilters(0.01, 0.001, 0.001, 5.0)
    ent = EntryCfg(entry_mode="maker", maker_timeout_s=300, maker_reprices=0, maker_on_timeout="skip")
    k = bars().copy()
    i = len(k) - 6
    lim = float(k["close"].iloc[i])
    k["low"] = lim * 1.001                                                # nothing trades through ...
    k["high"] = lim * 1.003
    k["open"] = lim * 1.002
    k["close"] = lim * 1.002
    k.loc[k.index[i], "close"] = lim
    b = ShadowBook(cfg)
    assert b.open("m", "A", "long", 0.9, lim, 1.0, None, f, entry=ent, decision_open_ms=int(k["open_time"].iloc[i]))
    assert b.has("m", "A") and not b.pos
    cut = k.iloc[:i + 1]
    assert b.on_bar("A", int(cut["open_time"].iloc[-1]), 0, 1e9, frame=cut) == [] and not b.pos      # no later bar yet
    cut = k.iloc[:i + 2]
    b.on_bar("A", int(cut["open_time"].iloc[-1]), float(cut["low"].iloc[-1]), float(cut["high"].iloc[-1]), frame=cut)
    assert not b.pos and not b.has("m", "A") and b.stats["m"]["dropped"] == 1      # timed out, skipped
    # now a bar trades through: fills AT the limit with the maker fee
    k.loc[k.index[i + 1], "low"] = lim * 0.995
    assert b.open("m", "A", "long", 0.9, lim, 1.0, None, f, entry=ent, decision_open_ms=int(k["open_time"].iloc[i]))
    cut = k.iloc[:i + 2]
    b.on_bar("A", int(cut["open_time"].iloc[-1]), float(cut["low"].iloc[-1]), float(cut["high"].iloc[-1]), frame=cut)
    p = b.pos["m|A"]
    assert p.entry == pytest.approx(lim) and p.entry_fee == pytest.approx(p.qty * lim * cfg.fee_maker)
    assert p.opened_at == (int(k["open_time"].iloc[i + 1]) + BAR) // 1000          # booked at the END of the fill bar
    assert b.stats["m"]["filled"] == 1


def test_shadow_maker_fill_replays_the_bars_after_the_fill_for_stops():
    cfg = Settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0}, allow_internal=True)
    f = SymbolFilters(0.01, 0.001, 0.001, 5.0)
    ent = EntryCfg(entry_mode="maker", maker_timeout_s=300, maker_reprices=0)
    k = bars(80).copy()
    i = len(k) - 8
    lim = float(k["close"].iloc[i])
    k["low"], k["high"], k["open"], k["close"] = lim * 1.001, lim * 1.003, lim * 1.002, lim * 1.002
    k.loc[k.index[i], "close"] = lim
    k.loc[k.index[i + 1], "low"] = lim * 0.995                           # fill bar
    k.loc[k.index[i + 3], "low"] = lim * 0.80                            # later crash: the stop must have hit
    b = ShadowBook(cfg)
    b.open("m", "A", "long", 0.9, lim, 1.0, None, f, entry=ent, decision_open_ms=int(k["open_time"].iloc[i]))
    out = b.on_bar("A", int(k["open_time"].iloc[-1]), float(k["low"].iloc[-1]), float(k["high"].iloc[-1]), frame=k)
    assert len(out) == 1 and out[0]["exit_reason"] == "sl" and not b.pos


# ------------------------------------------------------------------ cross-sectional models
SYMS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "BTCUSDT")
XM = dict(kind="xsec", history_bars_5m=100, rebalance_every_h=1, top_k=1, bottom_k=1, thresholds={},
          entry_cfg={"decision_every_bars": 12, "entry_mode": "taker"},
          exit_cfg={"exit_mode": "pct", "stop_vol_mult": 2.0, "stop_min": 0.02, "stop_max": 0.20})


def xsec_engine(store, tmp_path, doubles, pct, extra=None, challenger=False):
    d = tmp_path / "models"
    # (short history so the vol stop needs the fallback below)
    write_v3(d, "xs", "challenger" if challenger else "champion", **XM)
    if challenger:
        write_v3(d, "base", "champion", kind="mtf", history_bars_5m=100)
        doubles["base"] = V2("base", pl=0.0)                               # the champion stays out of the market
    doubles["xs"] = Ranker("xs", pct)
    eng, data = mk_engine(store, tmp_path, symbols=SYMS, now=hour_now(), settings={"short_enabled": True, **(extra or {})})
    # long history so the volatility-scaled stop can be computed
    now_ms = int(eng.clock() * 1000)
    for i, s in enumerate(SYMS):
        data.kl[s] = make_klines(2500, seed=i + 1, end_ms=now_ms)
        data.marks[s] = float(data.kl[s]["close"].iloc[-1])
    eng.universe = list(SYMS[:4])
    eng.update_hist = None
    eng.hist_need = max(eng.hist_need, 2300)
    slot = E.rebalance_slot(eng.now_ms(), 1)
    eng.xsec_slots["xs"] = slot - 1
    return eng, data, slot


def test_xsec_champion_opens_target_legs_with_stops_and_no_correlation_cap(store, tmp_path, doubles, monkeypatch):
    def no_corr(*a, **k):
        raise AssertionError("the correlation cap must not run inside a neutral book")

    monkeypatch.setattr(E, "correlation_gate", no_corr)
    eng, data, slot = xsec_engine(store, tmp_path, doubles, {"AAAUSDT": 0.9, "BBBUSDT": 0.6, "CCCUSDT": 0.4, "DDDUSDT": 0.1})
    run(eng.bars_cycle())
    pos = {p["symbol"]: p for p in store.get_positions()}
    assert set(pos) == {"AAAUSDT", "DDDUSDT"} and pos["AAAUSDT"]["side"] == "long" and pos["DDDUSDT"]["side"] == "short"
    for p in pos.values():
        assert p["sl_order_id"] and p["tp_order_id"] is None and p["tp_price"] is None and p["exit_mode"] == "pct"
        assert 0.02 - 1e-3 <= abs(p["entry_price"] - p["sl_price"]) / p["entry_price"] <= 0.20 + 1e-3
    assert eng.xsec_book == {"AAAUSDT": "long", "DDDUSDT": "short"}
    assert "BTCUSDT" in eng.challengers.get("xs", (eng.scorer,))[0].calls[-1] or "BTCUSDT" in eng.scorer.calls[-1]   # reference coin supplied
    assert eng.xsec_slots["xs"] == slot
    assert store.kv_get("xsec_book") == eng.xsec_book


def test_xsec_rebalance_closes_dropped_legs_keeps_unchanged_and_opens_new(store, tmp_path, doubles):
    eng, data, slot = xsec_engine(store, tmp_path, doubles, {"AAAUSDT": 0.9, "BBBUSDT": 0.6, "CCCUSDT": 0.4, "DDDUSDT": 0.1})
    run(eng.bars_cycle())
    ids = {p["symbol"]: p["id"] for p in store.get_positions()}
    eng.scorer.pct = {"AAAUSDT": 0.2, "BBBUSDT": 0.95, "CCCUSDT": 0.4, "DDDUSDT": 0.05}      # A drops out, B is the new long
    eng.xsec_slots["xs"] = slot - 1
    run(eng.universe_models_cycle())
    pos = {p["symbol"]: p for p in store.get_positions()}
    assert set(pos) == {"BBBUSDT", "DDDUSDT"} and pos["DDDUSDT"]["id"] == ids["DDDUSDT"]        # the short leg is untouched
    closed = [t for t in store.get_trades(10) if t["symbol"] == "AAAUSDT"]
    assert closed and closed[0]["exit_reason"] == "rebalance"
    assert eng.xsec_book == {"BBBUSDT": "long", "DDDUSDT": "short"}
    eng.xsec_slots["xs"] = slot                                                                  # same slot again: nothing happens
    n = len(store.get_trades(50))
    run(eng.universe_models_cycle())
    assert len(store.get_trades(50)) == n


def test_xsec_legs_obey_the_open_risk_cap(store, tmp_path, doubles):
    eng, data, slot = xsec_engine(store, tmp_path, doubles, {"AAAUSDT": 0.9, "BBBUSDT": 0.6, "CCCUSDT": 0.4, "DDDUSDT": 0.1},
                                  )
    store.patch_settings({"max_open_risk_pct": 12.0}, allow_internal=True)       # one leg risks ~10 % of equity
    eng.cfg = store.get_settings()
    run(eng.bars_cycle())
    assert len(store.get_positions()) == 1 and len(eng.xsec_book) == 1               # the cap binds: no overshoot
    b = eng.book()
    assert b.open_risk / eng.equity * 100 <= 12.0 + 1e-6


def test_xsec_rebalance_is_skipped_when_killed_or_degraded(store, tmp_path, doubles):
    eng, data, slot = xsec_engine(store, tmp_path, doubles, {"AAAUSDT": 0.9, "DDDUSDT": 0.1, "BBBUSDT": 0.5, "CCCUSDT": 0.5})
    eng.kill = {"tripped": True, "reason": "t", "t": 0}
    run(eng.bars_cycle())
    assert store.get_positions() == []
    eng.kill = {"tripped": False, "reason": "", "t": None}
    eng.safety_view = {"state": "DEGRADED", "reasons": []}
    eng.xsec_slots["xs"] = slot - 1
    run(eng.universe_models_cycle())
    assert store.get_positions() == []                                    # try_open obeys the guard state per leg


def test_xsec_first_sight_waits_for_the_next_boundary_and_stale_slots_are_missed(store, tmp_path, doubles):
    eng, data, slot = xsec_engine(store, tmp_path, doubles, {"AAAUSDT": 0.9, "DDDUSDT": 0.1, "BBBUSDT": 0.5, "CCCUSDT": 0.5})
    del eng.xsec_slots["xs"]
    run(eng.bars_cycle())
    assert store.get_positions() == [] and eng.xsec_slots["xs"] == slot
    eng.xsec_slots["xs"] = slot - 1
    eng.clock = lambda: hour_now() + 2000                                 # 33 min after the boundary: too late
    run(eng.universe_models_cycle())
    assert store.get_positions() == [] and eng.xsec_slots["xs"] == slot


def test_xsec_targets_ignore_untradable_and_need_short_enabled():
    sc = {"A": S.Score(0.9, 0.1), "B": S.Score(0.5, 0.5), "C": S.Score(0.1, 0.9), "Z": S.Score(0.0, 0.0)}
    t = E.Engine.xsec_targets(sc, XsecCfg(top_k=1, bottom_k=1), True)
    assert t == {"A": "long", "C": "short"}
    assert E.Engine.xsec_targets(sc, XsecCfg(top_k=2, bottom_k=2), False) == {"A": "long", "B": "long"}
    t = E.Engine.xsec_targets(sc, XsecCfg(top_k=1, bottom_k=5), True)
    assert set(t.values()) == {"long", "short"} and "Z" not in t and len(t) == 3


def test_xsec_challenger_shadows_a_neutral_book(store, tmp_path, doubles):
    eng, data, slot = xsec_engine(store, tmp_path, doubles, {"AAAUSDT": 0.9, "BBBUSDT": 0.6, "CCCUSDT": 0.4, "DDDUSDT": 0.1},
                                  challenger=True)
    run(eng.bars_cycle())
    legs = {p.symbol: p for p in eng.shadow.pos.values() if p.model == "xs"}
    assert set(legs) == {"AAAUSDT", "DDDUSDT"} and legs["AAAUSDT"].side == "long" and legs["DDDUSDT"].side == "short"
    assert all(p.tp is None for p in legs.values())                        # stop only
    eng.challengers["xs"][0].pct = {"AAAUSDT": 0.5, "BBBUSDT": 0.95, "CCCUSDT": 0.4, "DDDUSDT": 0.1}
    eng.xsec_slots["xs"] = slot - 1
    run(eng.universe_models_cycle())
    assert {p.symbol for p in eng.shadow.pos.values() if p.model == "xs"} == {"BBBUSDT", "DDDUSDT"}
    sh = store.get_shadow_trades("xs")
    assert sh and sh[0]["symbol"] == "AAAUSDT" and sh[0]["exit_reason"] == "rebalance"
    assert store.get_positions() == []                                     # a challenger never trades for real


# ------------------------------------------------------------------ safety / status
def test_paper_only_challenger_of_any_kind_is_refused_in_live_mode(store, tmp_path, doubles):
    d = tmp_path / "models"
    write_v3(d, "paper_only", "challenger", kind="gbm", note="PAPER TEST ONLY - nope")
    write_v3(d, "ok", "challenger", kind="mtf")
    eng, _ = mk_engine(store, tmp_path)
    eng.mode = "live"
    eng.challengers.clear()
    eng.reload_models(force=True)
    assert set(eng.challengers) == {"ok"}


def test_status_lists_models_decisions_and_maker_stats(store, tmp_path, doubles, monkeypatch):
    monkeypatch.setenv("TB_MODELS", str(tmp_path / "models"))
    d = tmp_path / "models"
    write_v3(d, "champ", "champion", kind="mtf", history_bars_5m=100, entry_cfg={"decision_every_bars": 12, "entry_mode": "maker"})
    write_v3(d, "xs", "challenger", **XM)
    eng, _ = mk_engine(store, tmp_path, now=hour_now())
    run(eng.heartbeat_cycle())
    st = store.kv_get("status")
    by = {m["name"]: m for m in st["models"]}
    assert by["champ"]["kind"] == "mtf" and by["champ"]["role"] == "champion" and by["champ"]["entry_mode"] == "maker"
    assert by["xs"]["kind"] == "xsec" and by["xs"]["rebalance_every_h"] == 1 and by["xs"]["next_decision_t"] > eng.clock()
    assert st["decisions"]["next_decision_t"] == pytest.approx(hour_now() - 10 + 3600) and st["decisions"]["decision_every_bars"] == 12
    assert st["maker"]["entries"] == 0 and st["maker"]["fill_rate"] is None and st["model"]["kind"] == "mtf"
    from fastapi.testclient import TestClient
    from tb.api import create_app

    c = TestClient(create_app(store.path, control_wait_s=0.0))
    api = c.get("/api/status").json()
    assert {m["name"] for m in api["models"]} == {"champ", "xs"} and api["maker"]["entries"] == 0
    sh = {r["model"]: r for r in c.get("/api/shadow").json()}
    assert sh["xs"]["kind"] == "xsec" and sh["xs"]["loaded"] and sh["champ"]["entry_mode"] == "maker"
