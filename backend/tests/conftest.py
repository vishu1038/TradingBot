from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tb.config import Settings  # noqa: E402
from tb.exchange.base import (Exchange, KLINE_COLUMNS, Premium, SymbolFilters, SymbolInfo, Ticker24h)  # noqa: E402
from tb.store import Store  # noqa: E402

BAR_MS = 300_000


def make_klines(n: int = 700, base: float = 100.0, seed: int = 1, end_ms: int | None = None,
                vol: float = 0.002) -> pd.DataFrame:
    """Synthetic 5m klines ending with the last bar open at the most recent 5m boundary."""
    rng = np.random.default_rng(seed)
    end_ms = end_ms or int(time.time() * 1000)
    last_open = end_ms // BAR_MS * BAR_MS
    t = last_open - np.arange(n - 1, -1, -1) * BAR_MS
    r = rng.normal(0, vol, n)
    c = base * np.exp(np.cumsum(r))
    o = np.concatenate([[base], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 3, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 3, n)))
    v = rng.uniform(100, 200, n)
    df = pd.DataFrame({"open_time": t.astype("int64"), "open": o, "high": h, "low": l, "close": c, "volume": v,
                       "quote_volume": v * c, "trades": np.full(n, 100, dtype="int64"),
                       "taker_buy_volume": v / 2, "taker_buy_quote_volume": v * c / 2})
    return df[KLINE_COLUMNS]


class FakeData(Exchange):
    """Scripted market-data source (stands in for Binance public endpoints)."""
    name = "fake"

    def __init__(self, symbols=("AAAUSDT", "BBBUSDT", "BTCUSDT"), base=100.0):
        self.kl = {s: make_klines(base=base, seed=i + 1) for i, s in enumerate(symbols)}
        self.marks = {s: float(self.kl[s]["close"].iloc[-1]) for s in symbols}
        self.fail = False

    def server_time_ms(self): return int(time.time() * 1000)

    def klines(self, symbol, interval="5m", limit=500, start_ms=None, end_ms=None):
        if self.fail:
            from tb.exchange.base import NetworkError
            raise NetworkError("boom")
        return self.kl[symbol].tail(limit).reset_index(drop=True)

    def tickers_24h(self): return [Ticker24h(s, m, 1e9, 1.0) for s, m in self.marks.items()]

    def exchange_info(self):
        return {s: SymbolInfo(s, filters=SymbolFilters(0.01, 0.001, 0.001, 5.0)) for s in self.marks}

    def premium(self, symbol=None):
        if self.fail:
            from tb.exchange.base import NetworkError
            raise NetworkError("boom")
        return [Premium(s, m, m, 0.0001, 0) for s, m in self.marks.items() if symbol in (None, s)]

    def open_interest(self, symbol): return 1000.0
    def open_interest_hist(self, symbol, period="5m", limit=13): return [(0, 1.0), (1, 1.1)]
    def long_short_ratio(self, symbol, period="5m"): return 1.2
    # trading methods are not used on the data source
    def set_leverage(self, *a): raise NotImplementedError
    def set_isolated(self, *a): raise NotImplementedError
    def market_order(self, *a, **k): raise NotImplementedError
    def place_stop(self, *a): raise NotImplementedError
    def place_take_profit(self, *a): raise NotImplementedError
    def cancel_order(self, *a): raise NotImplementedError
    def cancel_all(self, *a): raise NotImplementedError
    def open_conditional_orders(self, symbol=None): return []
    def positions(self): return []
    def balance(self): raise NotImplementedError
    def last_close(self, *a, **k): return None


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "tb.sqlite")
    yield s
    s.close()


@pytest.fixture
def cfg() -> Settings:
    return Settings()


@pytest.fixture
def fake_data() -> FakeData:
    return FakeData()


LEGACY_OFF = {"max_net_exposure_pct": 0.0, "reversal_z": 0.0, "min_tp_usd": 0.0, "min_sl_usd": 0.0}


def _set_init_defaults(cls, overrides):
    """Swap defaults of a dataclass ``__init__`` in place; returns the original tuple."""
    import dataclasses

    names = [f.name for f in dataclasses.fields(cls) if f.init]
    old = cls.__init__.__defaults__
    new = tuple(overrides.get(n, d) for n, d in zip(names, old))
    cls.__init__.__defaults__ = new
    return old


@pytest.fixture(autouse=True)
def _legacy_risk_defaults(request):
    """The reversal protection (net-exposure cap, reversal detector) and the dollar floor (2026-10-19) are new rules with their own
    tests. Every OTHER test exercises other rules with several same-side positions / tiny sizes, so for them these four defaults are
    off; a test that wants the shipped defaults is marked ``@pytest.mark.real_risk_defaults``."""
    if request.node.get_closest_marker("real_risk_defaults"):
        yield
        return
    from tb.sim.template import SimConfig

    o1 = _set_init_defaults(Settings, LEGACY_OFF)
    o2 = _set_init_defaults(SimConfig, LEGACY_OFF)
    try:
        yield
    finally:
        Settings.__init__.__defaults__ = o1
        SimConfig.__init__.__defaults__ = o2


def pytest_configure(config):
    config.addinivalue_line("markers", "real_risk_defaults: run with the shipped net-exposure / reversal / dollar-floor defaults")


def pytest_collection_finish(session):
    """Move everything imported during collection (pandas, numpy, the whole tb package ...) out of the GC's reach: a full
    collection over that heap takes ~50 ms and landed inside timing-sensitive tests (test_alerts non-blocking dispatch)."""
    import gc

    gc.collect()
    gc.freeze()
