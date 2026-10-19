"""tb.sim.model_backtest: numpy correlation == tb.risk.return_corr, exit lookups, caps and gates on synthetic bars."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tb.config import Settings
from tb.exits import ExitSpec
from tb.risk import return_corr
from tb.sim import model_backtest as mb

BAR = mb.BAR
T0 = 1_700_000_000_000 // BAR * BAR


def _kl(close: np.ndarray, spread: float = 0.001, t0: int = T0) -> pd.DataFrame:
    n = len(close)
    o = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open_time": t0 + np.arange(n, dtype="int64") * BAR, "open": o,
                         "high": np.maximum(o, close) * (1 + spread), "low": np.minimum(o, close) * (1 - spread), "close": close})


def _fu() -> pd.DataFrame:
    return pd.DataFrame({"calc_time": np.zeros(0, "int64"), "funding_rate": np.zeros(0)})


def _market(series: dict[str, np.ndarray]) -> mb.Market:
    return mb.Market({s: (_kl(c), _fu()) for s, c in series.items()})


SPEC = ExitSpec("pct", "5m", tp_pct=0.03, sl_pct=0.06, tp_enabled=True, horizon_bars=1152)


def test_numpy_corr_equals_return_corr():
    rng = np.random.default_rng(3)
    base = np.cumprod(1 + rng.normal(0, 0.002, 800))
    a = base * 100
    b = np.cumprod(1 + 0.7 * np.diff(np.r_[1, base]) / np.r_[1, base][:-1] + rng.normal(0, 0.001, 800)) * 50
    m = _market({"A": a, "B": b})
    t_dec = T0 + 700 * BAR
    ka, kb = _kl(a), _kl(b)
    ref = return_corr(ka[ka.open_time + BAR <= t_dec].set_index("open_time")["close"],
                      kb[kb.open_time + BAR <= t_dec].set_index("open_time")["close"], 288)
    assert abs(m.corr("A", "B", t_dec) - ref) < 1e-12


def test_exit_tp_sl_timeout_and_tie():
    c = np.full(60, 100.0)
    c[10] = 104.0                                # bar 10 reaches +3 % (high = 104.1)
    m = _market({"X": c})
    t_x, px, why, end = m.exit_of("X", "long", T0 + 5 * BAR, 100.0, 94.0, 103.0, 1152, 0.0002, 0.0005, "trigger")
    assert (why, px) == ("tp", 103.0) and t_x == T0 + 11 * BAR and not end
    # short with the same bars: the +4 % bar hits the 6 % stop? no (needs 106) -> runs to the data end
    t_x, px, why, end = m.exit_of("X", "short", T0 + 5 * BAR, 100.0, 106.0, 97.0, 1152, 0.0002, 0.0005, "trigger")
    assert why == "open_end" and end
    # time stop at the close of the horizon-th bar after entry
    t_x, px, why, end = m.exit_of("X", "long", T0 + 20 * BAR, 100.0, 94.0, 103.0, 10, 0.0002, 0.0005, "trigger")
    assert why == "timeout" and t_x == T0 + 31 * BAR and abs(px - 100.0 * (1 - 0.0002)) < 1e-9
    # a bar that touches both the stop and the TP: the stop wins
    c2 = np.full(40, 100.0)
    c2[8] = 110.0
    k = _kl(c2); k.loc[8, "low"] = 90.0
    m2 = mb.Market({"X": (k, _fu())})
    assert m2.exit_of("X", "long", T0 + 3 * BAR, 100.0, 94.0, 103.0, 1152, 0.0, 0.0, "trigger")[2] == "sl"


def test_simulate_gates_and_caps():
    cfg = Settings(max_side_risk_pct=60.0, max_open_risk_pct=35.0, capital_usd=200.0, sl_cooldown_bars=24, cooldown_bars=6)
    c = np.full(2000, 100.0)
    m = _market({f"S{i}": c.copy() for i in range(8)})
    bar = T0 + 400 * BAR
    cand = pd.DataFrame({"sym": [f"S{i}" for i in range(8)], "bar": bar, "side": "short", "prob": np.linspace(0.9, 0.6, 8), "kind": "regular"})
    t_end = T0 + 1990 * BAR
    tr = mb.simulate(cand, m, cfg, SPEC, mb.SimCfg(caps=True), t_end)
    # side-risk cap: 60 % of the 35 % budget of $200 = $42 -> at most 4 stop-outs of ~$10.4 at the same time
    assert len(tr) == 4 and [t["sym"] for t in tr] == ["S0", "S1", "S2", "S3"]
    tr2 = mb.simulate(cand, m, cfg, SPEC, mb.SimCfg(caps=False), t_end)
    assert len(tr2) == 8
    # per-coin: a second candidate for the same coin while it is open / cooling down is skipped
    twice = pd.concat([cand.iloc[:1], cand.iloc[:1].assign(bar=bar + 3 * BAR)], ignore_index=True)
    assert len(mb.simulate(twice, m, cfg, SPEC, mb.SimCfg(caps=False), t_end)) == 1


def test_pct_exit_economics():
    cfg = Settings()
    c = np.full(300, 100.0)
    c[60:] = 96.0                                # -4 %: no stop (6 %), no tp, only the 1152-bar time stop (data ends first)
    m = _market({"X": c})
    cand = pd.DataFrame({"sym": ["X"], "bar": [T0 + 50 * BAR], "side": ["long"], "prob": [0.7], "kind": ["regular"]})
    tr = mb.simulate(cand, m, cfg, SPEC, mb.SimCfg(caps=True), T0 + 290 * BAR)
    assert len(tr) == 1 and tr[0]["end"]
    # take-profit trade: +3 % of the slipped fill, net of fees on both legs
    c3 = np.full(300, 100.0); c3[60:] = 104.0
    tr = mb.simulate(cand, _market({"X": c3}), cfg, SPEC, mb.SimCfg(caps=True), T0 + 290 * BAR)[0]
    assert tr["reason"] == "tp"
    exp = tr["qty"] * tr["fill"] * 0.03 - tr["qty"] * tr["fill"] * 0.0005 - tr["qty"] * tr["px"] * 0.0005
    assert abs(tr["pnl"] - exp) < 1e-9
    assert abs(tr["notional"] - 160.0) < 1.0           # $20 x 8x slice (sl_usd 10 / 6 % = 166 > 160 -> capped by the slice)


def test_protection_is_inert_for_pct_3_6():
    assert mb.protection_inert(Settings(), SPEC)
