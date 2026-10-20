"""tb.rules: indicators, regimes, setups (no look-ahead, window independence), the ``rules`` scorer kind and research == live parity."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tb import rules as R
from tb.config import Settings
from tb.exchange.base import SymbolFilters
from tb.exits import ExitSpec, resolve_exit
from tb.features_mtf import sma_atr
from tb.model_rules import RulesScorer, load as load_rules
from tb.protect import Protect, next_stop, partial_level
from tb.reversal import lock_stop
from tb.scorer import FrameBundle, ModelEntry, load_model, score_frames, scan_models
from tb.shadow import ShadowBook

BAR = 300_000
T0 = 1_735_689_600_000          # 2025-01-01 00:00 UTC (hour aligned)


def synth(n: int = 20_000, seed: int = 1, drift_scale: float = 4e-4, t0: int = T0) -> pd.DataFrame:
    """Random-walk 5m klines with slowly alternating up / down trends, vol clusters and volume spikes."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    drift = drift_scale * np.sin(2 * np.pi * t / 6000.0 + seed) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 1700.0))
    vol = 0.002 * (1.0 + 0.6 * np.sin(2 * np.pi * t / 900.0 + 1.0))
    ret = drift + vol * rng.standard_normal(n)
    c = 100.0 * np.exp(np.cumsum(ret))
    o = np.concatenate([[100.0], c[:-1]])
    wick = np.abs(rng.standard_normal((2, n))) * vol * 0.6
    h = np.maximum(o, c) * (1 + wick[0])
    l = np.minimum(o, c) * (1 - wick[1])
    v = np.exp(rng.standard_normal(n) * 0.5 + 3.0) * (1 + 4.0 * (rng.random(n) < 0.02))
    return pd.DataFrame({"open_time": t0 + t * BAR, "open": o, "high": h, "low": l, "close": c, "volume": v,
                         "quote_volume": v * c})


VARIANTS = [R.RuleCfg(m, s, tf) for m in R.MODES for s in R.SETUPS for tf in R.TFS]


@pytest.fixture(scope="module")
def coin():
    return synth(20_000, seed=1)


@pytest.fixture(scope="module")
def btc():
    return synth(20_000, seed=2, drift_scale=6e-4)


@pytest.fixture(scope="module")
def sigs(coin, btc):
    return {v.key: R.rule_signals(coin, btc, v) for v in VARIANTS}


# ------------------------------------------------------------------------------------------ indicators
def test_ema_matches_pandas_and_is_finite_memory():
    x = np.cumsum(np.random.default_rng(0).standard_normal(3000)) + 100
    e = R.ema(x, 20)
    ref = pd.Series(x).ewm(span=20, adjust=False).mean().to_numpy()
    assert np.isnan(e[:R.ema_taps(20) - 1]).all()
    assert np.allclose(e[300:], ref[300:], rtol=0, atol=0.05 * x.std())           # tail weight e**-6 ~ 0.25 %
    # window independence: the value at t depends on the last K inputs only
    k = R.ema_taps(20)
    assert R.ema(x[-k:], 20)[-1] == pytest.approx(e[-1], rel=0, abs=0)
    assert np.allclose(R.ema(np.full(500, 7.0), 12)[100:], 7.0)


def test_rsi_bounds_directions_and_flat():
    up = np.linspace(100, 200, 400)
    assert R.rsi(up)[-1] == pytest.approx(100.0)
    assert R.rsi(up[::-1])[-1] == pytest.approx(0.0)
    assert R.rsi(np.full(400, 5.0))[-1] == 50.0
    x = 100 + np.cumsum(np.random.default_rng(3).standard_normal(2000))
    r = R.rsi(x)
    assert np.nanmin(r) >= 0 and np.nanmax(r) <= 100 and np.isnan(r[:R.rsi.__defaults__[0] * 6 - 1]).all()


def test_macd_histogram_sign_follows_momentum():
    up = 100 * np.exp(np.cumsum(np.full(600, 0.002)))
    line, sig, hist = R.macd(up)
    assert line[-1] > 0 and np.isfinite(hist[-1])
    flat_then_up = np.concatenate([np.full(300, 100.0), 100 * np.exp(np.cumsum(np.full(40, 0.01)))])
    assert R.macd(flat_then_up)[2][-1] > 0
    dn = up[::-1].copy()
    assert R.macd(dn)[0][-1] < 0


def test_volume_z_uses_only_the_previous_bars():
    v = np.random.default_rng(4).random(200) + 1
    v[100] = 50.0
    z = R.volume_z(v, 48)
    assert z[100] > 10
    z2 = R.volume_z(np.concatenate([v[:101], np.full(50, 1e9)]), 48)       # the future cannot move the past
    assert np.array_equal(z[:101], z2[:101], equal_nan=True)
    assert np.isnan(z[:48]).all() and np.isfinite(z[49])


def test_breakout_levels_exclude_the_current_bar():
    h = np.arange(30, dtype=float)
    l = h - 1
    hi, lo = R.breakout_levels(h, l, 20)
    assert hi[25] == 24.0 and lo[25] == 4.0 and np.isnan(hi[19]) and hi[20] == 19.0


def test_atr_sma_equals_the_exit_atr(coin):
    k = coin.iloc[:600]
    a = R.atr_sma(k["high"], k["low"], k["close"], 14)
    ref = sma_atr(k["high"].to_numpy(), k["low"].to_numpy(), k["close"].to_numpy(), 14)
    assert np.allclose(a[14:], ref[14:], equal_nan=True)


def test_bars_1h_keeps_complete_hours_only(coin):
    k = coin.iloc[5:5 + 12 * 10 + 7]                         # starts mid-hour, ends mid-hour
    b = R.bars_1h(k)
    assert len(b) == 10 and (b.t_end % R.HOUR_MS == 0).all()
    j = np.searchsorted(k["open_time"].to_numpy() + BAR, b.t_end[0]) 
    assert b.c[0] == k["close"].iloc[j] and b.o[0] == k["open"].iloc[j - 11]
    assert b.h[0] == k["high"].iloc[j - 11:j + 1].max() and b.v[0] == pytest.approx(k["volume"].iloc[j - 11:j + 1].sum())


# ------------------------------------------------------------------------------------------ regimes
def test_regimes_up_down_chop():
    n = 1500
    up = 100 * np.exp(np.cumsum(np.full(n, 0.002)))
    assert R.btc_regime(up)[-1] == 1 and R.btc_regime(up[::-1].copy())[-1] == -1
    assert R.coin_regime(up)[-1] == 1 and R.coin_regime(up[::-1].copy())[-1] == -1
    flat = 100 + np.sin(np.arange(n) / 5.0)
    assert R.btc_regime(flat)[-1] == 0
    assert (R.btc_regime(up)[:100] == 0).all()                # no history -> chop (no trades)
    assert R.coin_regime(up).dtype == np.int8


def test_regime_is_asof_the_bar_close(coin, btc):
    v = R.RuleCfg("btc", "S3", "5m")
    s = R.rule_signals(coin, btc, v)
    b1 = R.bars_1h(btc)
    reg = R.btc_regime(b1.c)
    tc = coin["open_time"].to_numpy() + BAR
    for i in (9000, 12345, 19999):
        j = np.searchsorted(b1.t_end, tc[i], side="right") - 1
        assert s.regime[i] == reg[j]
        assert b1.t_end[j] <= tc[i]


# ------------------------------------------------------------------------------------------ setups / no look-ahead
def test_variants_produce_signals_and_sides_follow_the_regime(sigs):
    for k, s in sigs.items():
        assert not (s.long & s.short).any(), k
        assert (s.regime[s.long] > 0).all() and (s.regime[s.short] < 0).all(), k
        assert (s.long | s.short)[s.due].sum() == (s.long | s.short).sum()
    assert sum(int((s.long | s.short).sum()) for s in sigs.values()) > 200
    assert all(int((s.long | s.short).sum()) > 0 for k, s in sigs.items() if k.endswith("5m"))
    # 1h variants decide on hour closes only
    for k, s in sigs.items():
        if k.endswith("1h"):
            tc = T0 + np.nonzero(s.long | s.short)[0] * BAR + BAR
            assert (tc % R.HOUR_MS == 0).all()


def test_setup_conditions_hold_at_every_signal(coin, sigs):
    c, h, l, v = (coin[k].to_numpy() for k in ("close", "high", "low", "volume"))
    # S2 (5m): close beyond the previous 20-bar extreme, volume z > 2, histogram on the same side
    s = sigs["coin_S2_5m"]
    hist = R.macd(c)[2]
    hi = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    lo = pd.Series(l).rolling(20).min().shift(1).to_numpy()
    for i in np.flatnonzero(s.long):
        assert c[i] > hi[i] and hist[i] > 0 and R.volume_z(v, 48)[i] > 2.0
    for i in np.flatnonzero(s.short):
        assert c[i] < lo[i] and hist[i] < 0
    # S3 (5m): MACD histogram crosses zero with RSI in the band
    s = sigs["btc_S3_5m"]
    rsi = R.rsi(c)
    for i in np.flatnonzero(s.long):
        assert hist[i] > 0 >= hist[i - 1] and 50 <= rsi[i] <= 70
    for i in np.flatnonzero(s.short):
        assert hist[i] < 0 <= hist[i - 1] and 30 <= rsi[i] <= 50
    # S1 (5m): pulled back to the EMA20 with RSI < 40 / > 60 within 6 bars, histogram turned
    s = sigs["coin_S1_5m"]
    e20 = R.ema(c, 20)
    for i in np.flatnonzero(s.long):
        assert (l[i - 5:i + 1] <= e20[i - 5:i + 1]).any() and rsi[i - 5:i + 1].min() < 40
        assert hist[i] > hist[i - 1] and hist[i - 1] <= hist[i - 2]
    for i in np.flatnonzero(s.short):
        assert (h[i - 5:i + 1] >= e20[i - 5:i + 1]).any() and rsi[i - 5:i + 1].max() > 60
        assert hist[i] < hist[i - 1] and hist[i - 1] >= hist[i - 2]


def test_no_lookahead_prefix_equals_full(coin, btc, sigs):
    """A signal at row i is the same whether or not the rows after i exist."""
    cut = 15_000
    for v in VARIANTS:
        pre = R.rule_signals(coin.iloc[:cut + 1].reset_index(drop=True), btc.iloc[:cut + 1].reset_index(drop=True), v)
        full = sigs[v.key]
        assert np.array_equal(pre.long, full.long[:cut + 1]), v.key
        assert np.array_equal(pre.short, full.short[:cut + 1]), v.key
        assert np.array_equal(pre.regime, full.regime[:cut + 1]), v.key


def test_window_independence_tail_equals_full(coin, btc, sigs):
    """The live scorer sees only the last ``history_bars_5m`` rows and reads the newest one: it equals the 2-year research
    array, bit for bit (checked at signal rows and at random rows, every variant)."""
    rng = np.random.default_rng(9)
    for v in VARIANTS:
        n = v.history_bars_5m()
        assert n <= 9600
        full = sigs[v.key]
        hit = np.flatnonzero(full.long | full.short)
        rows = [i for i in list(rng.choice(hit, size=min(8, len(hit)), replace=False)) + list(rng.integers(n, len(coin), size=12))
                if i + 1 >= n]
        for i in rows:
            tail = R.rule_signals(coin.iloc[i + 1 - n:i + 1].reset_index(drop=True), btc.iloc[i + 1 - n:i + 1].reset_index(drop=True), v)
            assert (bool(tail.long[-1]), bool(tail.short[-1]), int(tail.regime[-1])) == \
                   (bool(full.long[i]), bool(full.short[i]), int(full.regime[i])), (v.key, int(i))


def test_missing_btc_or_short_history_means_no_trades(coin):
    s = R.rule_signals(coin, None, R.RuleCfg("btc", "S3", "5m"))
    assert not s.long.any() and not s.short.any()
    s = R.rule_signals(coin.iloc[:3000], coin.iloc[:3000], R.RuleCfg("coin", "S3", "5m"))      # < 30 days: regimes still chop
    assert not s.long.any() and not s.short.any()
    e = R.rule_signals(coin.iloc[:0], None, R.RuleCfg("coin", "S1", "1h"))
    assert len(e.long) == 0


def test_rulecfg_roundtrip_and_validation():
    c = R.RuleCfg("coin", "S2", "1h")
    assert R.RuleCfg.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    assert c.key == "coin_S2_1h"
    with pytest.raises(ValueError):
        R.RuleCfg("coin", "S9", "1h")
    assert 8000 <= R.RuleCfg("btc", "S1", "5m").history_bars_5m() <= 9600


# ------------------------------------------------------------------------------------------ scorer kind + parity
def artifact(v: R.RuleCfg, d: Path, status: str = "challenger") -> Path:
    meta = {"name": f"rules_{v.key}_test", "version": "t", "created_at": "2026-10-20T00:00:00+00:00", "status": status,
            "kind": "rules", "rule_cfg": v.to_dict(), "timeframes": ["5m", "1h"], "history_bars_5m": v.history_bars_5m(),
            "thresholds": {"long": 0.5, "short": 0.5},
            "exit_cfg": {"exit_mode": "atr", "atr_tf": "1h", "tp_atr": 1.5, "sl_atr": 3.0, "horizon_bars": 1152},
            "entry_cfg": {"decision_every_bars": 12 if v.tf == "1h" else 1, "cooldown_bars": 6},
            "note": "PAPER TEST ONLY - test"}
    p = d / f"{meta['name']}.json"
    p.write_text(json.dumps(meta))
    return p


def test_scorer_kind_loads_through_the_registry(tmp_path):
    v = R.RuleCfg("btc", "S3", "5m")
    p = artifact(v, tmp_path)
    entries = scan_models(tmp_path)
    assert len(entries) == 1 and entries[0].status == "challenger"
    got = load_model(entries[0])
    assert got is not None
    sc, info = got
    assert isinstance(sc, RulesScorer) and info.kind == "rules" and info.history_bars_5m == v.history_bars_5m()
    assert info.exit_cfg["atr_tf"] == "1h" and info.threshold_long == 0.5
    spec = resolve_exit(info.exit_cfg, info.kind, info.name)
    assert spec.mode == "atr" and spec.atr_tf == "1h" and spec.tp_atr == 1.5 and spec.sl_atr == 3.0 and spec.horizon_bars == 1152
    assert load_rules(p).cfg == v


@pytest.mark.parametrize("v", VARIANTS, ids=lambda v: v.key)
def test_live_scorer_equals_research_arrays(v, coin, btc, sigs, tmp_path):
    """Research == live: on the bars the engine would hand over (closed rows up to bar i, history cut to the model's need) the
    scorer outputs exactly the research signal of row i."""
    artifact(v, tmp_path)
    sc, info = load_model(scan_models(tmp_path)[0])
    full = sigs[v.key]
    idx = np.flatnonzero(full.long | full.short)
    assert len(idx) > 0
    rng = np.random.default_rng(5)
    pick = list(idx[rng.choice(len(idx), size=min(6, len(idx)), replace=False)])
    pick += list(rng.integers(9000, len(coin) - 1, size=10)) + [len(coin) - 1]
    hist = info.history_bars_5m
    for i in pick:
        lo = max(0, i + 1 - 12_000)                           # the engine holds MORE bars than the model needs
        fb = FrameBundle("ETHUSDT", coin.iloc[lo:i + 1].reset_index(drop=True), btc_kl=btc.iloc[lo:i + 1].reset_index(drop=True))
        s = score_frames(sc, info, {"ETHUSDT": fb})["ETHUSDT"]
        want = (1.0, 0.0) if full.long[i] else (0.0, 1.0) if full.short[i] else (0.0, 0.0)
        assert (s.prob_long, s.prob_short) == want, (v.key, int(i))
        if want != (0.0, 0.0):
            assert s.reasons and s.reasons[0].feature == "regime"
    assert hist <= 9600


def test_btcusdt_uses_its_own_bars_and_missing_btc_is_silent(tmp_path, btc):
    v = R.RuleCfg("btc", "S3", "5m")
    artifact(v, tmp_path)
    sc, info = load_model(scan_models(tmp_path)[0])
    own = R.rule_signals(btc, btc, v)
    i = int(np.flatnonzero(own.long | own.short)[-1])
    fb = FrameBundle("BTCUSDT", btc.iloc[:i + 1].reset_index(drop=True), btc_kl=None)
    s = score_frames(sc, info, {"BTCUSDT": fb})["BTCUSDT"]
    assert (s.prob_long, s.prob_short) == ((1.0, 0.0) if own.long[i] else (0.0, 1.0))
    fb2 = FrameBundle("ETHUSDT", btc.iloc[:i + 1].reset_index(drop=True), btc_kl=None)
    s2 = score_frames(sc, info, {"ETHUSDT": fb2})["ETHUSDT"]
    assert (s2.prob_long, s2.prob_short) == (0.0, 0.0)


def test_1h_scorer_is_silent_between_hour_closes(coin, btc, sigs, tmp_path):
    v = R.RuleCfg("coin", "S2", "1h")
    artifact(v, tmp_path)
    sc, info = load_model(scan_models(tmp_path)[0])
    full = sigs[v.key]
    i = int(np.flatnonzero(full.long | full.short)[3])
    for off in (1, 2, 5):                                     # bars that do not close an hour
        fb = FrameBundle("X", coin.iloc[:i + off + 1].reset_index(drop=True), btc_kl=btc.iloc[:i + off + 1].reset_index(drop=True))
        s = score_frames(sc, info, {"X": fb})["X"]
        assert (s.prob_long, s.prob_short) == (0.0, 0.0)


# ------------------------------------------------------------------------------------------ profit protection cannot undo the TP
SPEC = ExitSpec("atr", "1h", tp_atr=1.5, sl_atr=3.0, horizon_bars=1152)
PROTECTIONS = [Protect(), Protect(breakeven_r=1.0, trail_enabled=True, trail_activate_r=1.0, trail_distance_atr=1.0,
                       partial_tp_r=1.0, partial_tp_frac=0.5, min_tp_usd=2.0),
               Protect(breakeven_r=0.4, trail_enabled=True, trail_activate_r=0.3, partial_tp_r=0.3)]


@pytest.mark.parametrize("side,d", [("long", 1.0), ("short", -1.0)])
def test_default_protection_is_inert_before_the_tp(side, d):
    """TP = 0.5 R (1.5 ATR of a 3 ATR stop): break-even (1 R), trail (1 R / 1.5 R) and the partial (1 R / 1.5 R) all sit beyond it."""
    entry, atr = 100.0, 2.0
    sl0, tp = entry - d * 3 * atr, entry + d * 1.5 * atr
    for p in PROTECTIONS[:2]:
        assert partial_level(side, entry, sl0, tp, p) is None
        best = tp - d * 1e-9                                      # the most favourable price an OPEN position can have seen
        act, sl, why = next_stop(side, entry, sl0, sl0, best, atr, False, p)
        assert why is None and sl == sl0 and not act
    # reversal protection's lock only ever moves the stop towards profit (never into a loss) and never replaces the TP
    lk = lock_stop(side, entry, sl0, sl0, entry + d * 1.0 * atr, 0.0005, 0.5)
    assert lk is None or (lk - sl0) * d > 0


def test_research_helper_asserts_inertness_from_the_real_functions():
    from tb.sim.rules_research import protection_inert, SPEC as RSPEC
    assert protection_inert(Settings(), RSPEC)


@pytest.mark.parametrize("side", ["long", "short"])
def test_shadow_trade_that_reaches_the_tp_never_ends_at_the_sl(side):
    """Shadow book with the rule models' exit: wild bars that run through TP, break-even and the trail level and come back
    still book the TP; the TP is evaluated before the protection ratchet on every bar."""
    cfg = Settings().merge({"slippage_pct": 0.0, "stop_slippage_pct": 0.0, "max_side_risk_pct": 100,
                            "breakeven_r": 1.0, "partial_tp_r": 1.0, "partial_tp_frac": 0.5,
                            "trail": {"enabled": True, "activate_r": 1.0, "distance_atr": 1.0}}, allow_internal=True)
    book = ShadowBook(cfg)
    f = SymbolFilters(0.01, 0.001, 0.001, 5.0)
    d = 1.0 if side == "long" else -1.0
    atr = 2.0
    assert book.open("m", "X", side, 1.0, 100.0, 1000, atr, f, spec=SPEC)
    p = book.pos["m|X"]
    assert p.tp == pytest.approx(100.0 + d * 3.0) and p.sl == pytest.approx(100.0 - d * 6.0)
    # one bar: touches 0.2R (no exit), next bar: swings through 2.5R and back below the entry -> TP first (it was reached)
    out = book.on_bar("X", 2000 * 1000, 99.0 if side == "long" else 100.8, 100.8 if side == "long" else 99.0, 2300.0)
    assert out == []
    if side == "long":
        lo, hi = 95.0, 112.0
    else:
        lo, hi = 88.0, 105.0
    out = book.on_bar("X", 2300 * 1000, lo, hi, 2600.0)
    assert len(out) == 1 and out[0]["exit_reason"] == "tp" and out[0]["pnl"] > 0
    # live-price path (mark ticks) behaves the same: a tick at the TP closes it before any stop move
    book.open("m", "Y", side, 1.0, 100.0, 5000, atr, f, spec=SPEC)
    res = book.on_price("Y", 100.0 + d * 3.0, 5100.0)
    assert len(res) == 1 and res[0]["exit_reason"] == "tp"
