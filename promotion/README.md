# Promotion gate — "ready for real money?"

`promotion/evaluator.py` is a self-assessment engine. It looks at the bot's
paper-trading track record and decides whether the strategy has earned the right
to start trading with real capital — and, just as importantly, explains *why* in
plain language.

> **This gate is necessary but NOT sufficient.** Passing every criterion means the
> strategy looked good *on paper*. Live trading differs: slippage, latency,
> partial/rejected fills, funding, exchange outages, and your own nerves under a
> real drawdown are not captured by a simulated record. A passing verdict is
> permission to start a **reduced-size live trial** (a small slice of intended
> capital), whose results you then re-assess — it is **not** permission to deploy
> full size, and it should never auto-enable live trading without a human in the
> loop.

## The criteria

All thresholds live in the `PromotionCriteria` dataclass. Every one of them is a
**hard gate**: `ready` is `True` only when *all* applicable criteria pass.

| Criterion | Default | Why |
|---|---|---|
| `min_trades` | `100` | Enough closed trades that the statistics aren't noise. |
| `min_days` | `14` | Edge is sustained across time/regimes, not one lucky day. |
| `min_total_return_pct` | `5.0` | The bot must actually make money. |
| `min_sharpe` | `1.0` | Return *per unit of risk*; ~1.0 is a common "worth trading" bar. |
| `max_drawdown_pct` | `15.0` | Measured drawdown must be shallower than -15%. |
| `min_win_rate_pct` | `45.0` | Fraction of profitable bars/trades (paired with profit factor). |
| `min_profit_factor` | `1.2` | Gross profit / gross loss; >1.0 is net profitable, 1.2 adds margin. |
| `consistency_min_positive_windows` | `0.6` | ≥60% of rolling sub-windows profitable — guards against one giant win masking many losses. |
| `n_consistency_windows` | `10` | How many sequential windows the record is split into for the consistency check. |

The `score` (0..1) is a **weighted** fraction of criteria passed — partial credit
so you can watch progress while still short of the bar. Sharpe and max-drawdown
carry the most weight (see `_WEIGHTS`); the score does not change the `ready`
decision, which always requires a clean sweep.

## How to use it

```python
from promotion.evaluator import PromotionEvaluator, PromotionCriteria

evaluator = PromotionEvaluator()  # or PromotionEvaluator(PromotionCriteria(min_sharpe=1.5))

# Preferred: from raw series (enables the consistency check).
verdict = evaluator.assess(
    equity,            # pd.Series of equity over time
    returns,           # pd.Series of per-bar returns
    n_trades=142,
    days_running=18,
    timeframe="1h",    # used to annualize Sharpe
)

# Or, when you only have a summarize()-style metrics dict:
verdict = evaluator.assess_metrics(metrics, n_trades=142, days_running=18)

print(verdict.ready, verdict.score)
print(evaluator.explain(verdict))      # full multi-line report
```

`verdict` is a `PromotionVerdict` with: `ready`, `score`, `passed`, `failed`,
`reasons`, `metrics`, `recommendation`, plus `.as_dict()` for handing to
`alerts.notifier.notify_promotion(notifier, verdict.as_dict())`.

`assess_metrics` skips the consistency check when the raw return series isn't
available (it says so in `reasons`); pass a `consistency` and/or `profit_factor`
key in the metrics dict if you've precomputed them.

## How to tune

Edit `PromotionCriteria` defaults, or construct one with overrides and pass it to
`PromotionEvaluator(criteria=...)`. Guidance:

- **Want a higher bar before risking money?** Raise `min_sharpe`, raise
  `min_trades`/`min_days`, lower `max_drawdown_pct`, raise `min_profit_factor`.
- **Strategy is low win-rate / high payoff** (trend following): lower
  `min_win_rate_pct` but keep `min_profit_factor` firm — that pair is what proves
  the math works.
- **Short timeframe with thousands of bars?** `min_trades`/`min_days` may pass
  trivially; lean on Sharpe, drawdown, and consistency instead.
- **Reweighting:** edit `_WEIGHTS` in `evaluator.py` to change how the partial
  `score` is computed (it does not affect the pass/fail `ready` decision).
