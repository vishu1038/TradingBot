"""Promotion gate — "is the bot good enough to switch from paper to real money?"

This is a self-assessment engine. Given a paper-trading track record (equity curve,
per-bar returns, trade count, days running) it evaluates a set of conservative,
explicit criteria and produces a `PromotionVerdict` that says *yes/no* and, just as
importantly, *why*. The bot can read this verdict, log it, and alert a human.

IMPORTANT — this gate is NECESSARY BUT NOT SUFFICIENT.
Passing every criterion here means the *strategy* looked good on paper. It does NOT
mean it will be profitable live: real execution differs from simulation in ways the
paper record cannot capture — slippage, latency, partial/rejected fills, funding,
exchange downtime, and your own behaviour under real drawdown. Treat a passing verdict
as permission to begin a REDUCED-SIZE LIVE TRIAL (a small fraction of intended capital)
whose results are then re-assessed, NOT as permission to deploy full size. Never wire
this verdict to auto-enable live trading without a human in the loop.

Tuning lives entirely in `PromotionCriteria`; see promotion/README.md.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.metrics import max_drawdown, sharpe_ratio

logger = logging.getLogger()

# Bars per year per timeframe, used to annualize Sharpe. Mirrors backtesting.metrics.
_PERIODS_PER_YEAR = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "4h": 2_190, "1d": 365,
}

# Relative weight of each criterion in the 0..1 score. All are "hard" gates for the
# `ready` boolean; weights only shape the partial-credit score shown while not ready.
_WEIGHTS = {
    "min_trades": 1.0,
    "min_days": 1.0,
    "min_total_return_pct": 1.5,
    "min_sharpe": 2.0,
    "max_drawdown_pct": 2.0,
    "min_win_rate_pct": 1.0,
    "min_profit_factor": 1.5,
    "consistency": 1.0,
}


@dataclass
class PromotionCriteria:
    """Thresholds the paper record must clear before going live.

    Defaults are deliberately conservative — better to keep paper trading a few
    extra weeks than to risk real money on a fluke. Tune in one place; every
    threshold maps to exactly one check in `PromotionEvaluator.assess`.
    """

    min_trades: int = 100
    """Minimum number of closed trades. Too few and the stats are just noise."""

    min_days: float = 14
    """Minimum wall-clock days the record must span — proves the edge is
    sustained across regimes, not a single lucky session."""

    min_total_return_pct: float = 5.0
    """Minimum cumulative return (%) over the whole record. The bot must
    actually make money, not merely avoid losing it."""

    min_sharpe: float = 1.0
    """Minimum annualized Sharpe. Rewards return *per unit of risk*; 1.0 is a
    common "worth trading" bar."""

    max_drawdown_pct: float = 15.0
    """Worst peak-to-trough decline allowed, as a positive percent. Measured
    drawdown must be shallower than this (i.e. > -15%)."""

    min_win_rate_pct: float = 45.0
    """Minimum fraction (%) of profitable bars/trades. A low win rate can still
    be fine with big winners, so this is paired with profit factor."""

    min_profit_factor: float = 1.2
    """Minimum gross profit / gross loss. > 1.0 means net profitable; 1.2 gives
    a margin of safety against live cost drift."""

    consistency_min_positive_windows: float = 0.6
    """Fraction of rolling sub-windows that must be profitable (>= 0.6 = 60%).
    Guards against one giant win masking a string of losing periods."""

    n_consistency_windows: int = 10
    """How many sequential windows to split the record into for the consistency
    check. ~10 is enough to detect lumpiness without becoming noisy."""


@dataclass
class PromotionVerdict:
    """Outcome of an assessment. `ready` is the bottom line; everything else
    explains it."""

    ready: bool
    score: float
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    recommendation: str = ""

    def as_dict(self) -> dict:
        """Plain dict form (e.g. for `alerts.notifier.notify_promotion`)."""
        return {
            "ready": self.ready,
            "score": self.score,
            "passed": self.passed,
            "failed": self.failed,
            "reasons": self.reasons,
            "metrics": self.metrics,
            "recommendation": self.recommendation,
        }


def _profit_factor(returns: pd.Series) -> float | None:
    """Gross gains / gross losses. None if there are no losses (undefined)."""
    r = returns.dropna()
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses == 0:
        return None
    return gains / losses


def _consistency(returns: pd.Series, n_windows: int) -> float | None:
    """Fraction of sequential windows with a positive summed return. None if too
    short to split meaningfully."""
    r = returns.dropna()
    if len(r) < n_windows or n_windows < 2:
        return None
    chunks = np.array_split(r.to_numpy(), n_windows)
    positive = sum(1 for c in chunks if c.sum() > 0)
    return positive / len(chunks)


class PromotionEvaluator:
    """Evaluates a paper-trading record against `PromotionCriteria`."""

    def __init__(self, criteria: PromotionCriteria = None) -> None:
        self.criteria = criteria or PromotionCriteria()

    def assess(
        self,
        equity: pd.Series,
        returns: pd.Series,
        *,
        n_trades: int,
        days_running: float,
        timeframe: str = "1h",
        profit_factor: float | None = None,
    ) -> PromotionVerdict:
        """Full assessment from raw series. Computes every metric and checks
        every criterion."""
        ppy = _PERIODS_PER_YEAR.get(timeframe, 8_760)
        returns = returns.dropna()

        sharpe = sharpe_ratio(returns, ppy)
        dd_pct = -max_drawdown(equity) * 100.0  # positive percent
        total_return_pct = (
            float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)
            if len(equity) >= 2
            else 0.0
        )
        wins = int((returns > 0).sum())
        losses = int((returns < 0).sum())
        win_rate_pct = (wins / (wins + losses) * 100.0) if (wins + losses) else 0.0
        if profit_factor is None:
            profit_factor = _profit_factor(returns)
        consistency = _consistency(returns, self.criteria.n_consistency_windows)

        metrics = {
            "total_return_pct": round(total_return_pct, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(dd_pct, 2),
            "win_rate_pct": round(win_rate_pct, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
            "consistency": round(consistency, 2) if consistency is not None else None,
            "trades": int(n_trades),
            "days_running": round(float(days_running), 1),
            "timeframe": timeframe,
        }
        return self._evaluate(metrics, consistency_available=consistency is not None)

    def assess_metrics(
        self,
        metrics: dict,
        *,
        n_trades: int,
        days_running: float,
    ) -> PromotionVerdict:
        """Assess directly from a `summarize()`-style metrics dict when the raw
        series aren't handy. The consistency check is skipped (and noted) unless
        a 'consistency' key is already present."""
        c = self.criteria
        pf = metrics.get("profit_factor")
        consistency = metrics.get("consistency")
        m = {
            "total_return_pct": metrics.get("total_return_pct", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            # Accept either positive or negative-signed drawdown; normalize to positive %.
            "max_drawdown_pct": abs(metrics.get("max_drawdown_pct", 0.0)),
            "win_rate_pct": metrics.get("win_rate_pct", 0.0),
            "profit_factor": pf,
            "consistency": consistency,
            "trades": int(n_trades),
            "days_running": round(float(days_running), 1),
            "timeframe": metrics.get("timeframe", "n/a"),
        }
        return self._evaluate(
            m,
            consistency_available=consistency is not None,
            pf_available=pf is not None,
        )

    def _evaluate(
        self,
        metrics: dict,
        *,
        consistency_available: bool,
        pf_available: bool = True,
    ) -> PromotionVerdict:
        """Shared scoring logic. Applies each criterion, builds passed/failed/
        reasons, and produces the verdict."""
        c = self.criteria
        passed: list[str] = []
        failed: list[str] = []
        reasons: list[str] = []
        earned = 0.0
        possible = 0.0

        def check(key: str, ok: bool, label: str) -> None:
            nonlocal earned, possible
            weight = _WEIGHTS.get(key, 1.0)
            possible += weight
            if ok:
                earned += weight
                passed.append(key)
                reasons.append(f"PASS  {label}")
            else:
                failed.append(key)
                reasons.append(f"FAIL  {label}")

        check(
            "min_trades",
            metrics["trades"] >= c.min_trades,
            f"trades {metrics['trades']} >= {c.min_trades}",
        )
        check(
            "min_days",
            metrics["days_running"] >= c.min_days,
            f"days {metrics['days_running']} >= {c.min_days}",
        )
        check(
            "min_total_return_pct",
            metrics["total_return_pct"] >= c.min_total_return_pct,
            f"total return {metrics['total_return_pct']}% >= {c.min_total_return_pct}%",
        )
        check(
            "min_sharpe",
            metrics["sharpe"] >= c.min_sharpe,
            f"Sharpe {metrics['sharpe']} >= {c.min_sharpe}",
        )
        check(
            "max_drawdown_pct",
            metrics["max_drawdown_pct"] <= c.max_drawdown_pct,
            f"max drawdown {metrics['max_drawdown_pct']}% <= {c.max_drawdown_pct}%",
        )
        check(
            "min_win_rate_pct",
            metrics["win_rate_pct"] >= c.min_win_rate_pct,
            f"win rate {metrics['win_rate_pct']}% >= {c.min_win_rate_pct}%",
        )

        # Profit factor: if undefined (no losses), treat as a pass — having zero
        # losing periods is not a reason to block.
        pf = metrics.get("profit_factor")
        if not pf_available or pf is None:
            check("min_profit_factor", True, "profit factor: no losses recorded (n/a)")
        else:
            check(
                "min_profit_factor",
                pf >= c.min_profit_factor,
                f"profit factor {pf} >= {c.min_profit_factor}",
            )

        # Consistency: only counts toward the gate when we could compute it.
        if consistency_available and metrics.get("consistency") is not None:
            check(
                "consistency",
                metrics["consistency"] >= c.consistency_min_positive_windows,
                f"positive windows {metrics['consistency']} "
                f">= {c.consistency_min_positive_windows}",
            )
        else:
            reasons.append(
                "SKIP  consistency check (raw return series unavailable)"
            )

        score = earned / possible if possible else 0.0
        ready = len(failed) == 0 and len(passed) > 0

        if ready:
            recommendation = (
                f"READY: all criteria met over {metrics['days_running']} days / "
                f"{metrics['trades']} trades. Begin a REDUCED-SIZE live trial "
                f"(small fraction of capital), then re-assess — paper success does "
                f"not guarantee live results."
            )
        else:
            top = self._top_failure(metrics)
            recommendation = (
                f"Continue paper trading: {top} "
                f"({len(failed)} of {len(passed) + len(failed)} criteria failing, "
                f"score {score:.0%})."
            )

        return PromotionVerdict(
            ready=ready,
            score=round(score, 4),
            passed=passed,
            failed=failed,
            reasons=reasons,
            metrics=metrics,
            recommendation=recommendation,
        )

    def _top_failure(self, metrics: dict) -> str:
        """Human phrasing for the single most important failing criterion (used
        in the recommendation line). Ordered by how much it should block going
        live."""
        c = self.criteria
        if metrics["trades"] < c.min_trades:
            return f"only {metrics['trades']} trades < {c.min_trades} (insufficient sample)"
        if metrics["days_running"] < c.min_days:
            return f"only {metrics['days_running']} days < {c.min_days} (too short)"
        if metrics["sharpe"] < c.min_sharpe:
            return f"Sharpe {metrics['sharpe']} < {c.min_sharpe} (risk-adjusted return too low)"
        if metrics["max_drawdown_pct"] > c.max_drawdown_pct:
            return (
                f"drawdown {metrics['max_drawdown_pct']}% exceeds "
                f"{c.max_drawdown_pct}% limit"
            )
        if metrics["total_return_pct"] < c.min_total_return_pct:
            return f"return {metrics['total_return_pct']}% < {c.min_total_return_pct}%"
        pf = metrics.get("profit_factor")
        if pf is not None and pf < c.min_profit_factor:
            return f"profit factor {pf} < {c.min_profit_factor}"
        if metrics["win_rate_pct"] < c.min_win_rate_pct:
            return f"win rate {metrics['win_rate_pct']}% < {c.min_win_rate_pct}%"
        consistency = metrics.get("consistency")
        if consistency is not None and consistency < c.consistency_min_positive_windows:
            return (
                f"only {consistency:.0%} of windows profitable < "
                f"{c.consistency_min_positive_windows:.0%}"
            )
        return "one or more criteria not met"

    def explain(self, verdict: PromotionVerdict) -> str:
        """Multi-line human-readable report of a verdict."""
        lines = []
        banner = "READY FOR LIVE" if verdict.ready else "NOT READY"
        lines.append("=" * 60)
        lines.append(f"  PROMOTION ASSESSMENT: {banner}")
        lines.append(f"  Score: {verdict.score:.0%} "
                     f"({len(verdict.passed)} passed, {len(verdict.failed)} failed)")
        lines.append("=" * 60)
        lines.append("Metrics:")
        for k, v in verdict.metrics.items():
            lines.append(f"  {k:>18}: {v}")
        lines.append("")
        lines.append("Criteria:")
        for r in verdict.reasons:
            lines.append(f"  {r}")
        lines.append("")
        lines.append(f"Recommendation: {verdict.recommendation}")
        if verdict.ready:
            lines.append("")
            lines.append(
                "REMINDER: this gate is necessary but NOT sufficient. Slippage, "
                "latency and real fills differ from paper. Start small."
            )
        lines.append("=" * 60)
        return "\n".join(lines)
