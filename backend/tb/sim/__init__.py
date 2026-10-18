"""Event-driven backtester (5m bars) implementing the shared trade template."""
from .template import SimConfig, breakeven_winrate  # noqa: F401
from .signals import always_long_every_n, random_entries  # noqa: F401
from .backtest import BacktestResult, run_backtest  # noqa: F401
