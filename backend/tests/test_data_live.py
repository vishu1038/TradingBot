"""REST top-up: parsing, metric timestamp mapping, closed-bars-only, paging, merge (offline fake API)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from tb.data import frames, live
from tb.data.paths import BAR_MS, KLINE_COLS, METRIC_COLS

T0 = 1_791_590_400_000          # 2026-10-10 00:00 UTC
NOW = T0 + 100 * BAR_MS + 17_000  # 17 s into the bar that opens at T0 + 100 bars


class FakeAPI:
    def __init__(self):
        self.calls = []

    def __call__(self, path, params):
        self.calls.append((path, dict(params)))
        if path == "/fapi/v1/klines":
            out, t = [], params["startTime"]
            while t <= params["endTime"] and len(out) < params["limit"] and t <= T0 + 200 * BAR_MS:
                out.append([t, "100", "101", "99", "100.5", "10", t + BAR_MS - 1, "1005", 7, "4", "402", "0"])
                t += BAR_MS
            return out
        if path == "/fapi/v1/fundingRate":
            return [{"symbol": "X", "fundingTime": T0 + 28_800_000 * i + 2, "fundingRate": "0.0001", "markPrice": "1"}
                    for i in range(3) if T0 + 28_800_000 * i + 2 >= params["startTime"]]
        ep = path.rsplit("/", 1)[1]
        ts = [t for t in range(T0 + BAR_MS, T0 + 400 * BAR_MS, BAR_MS)
              if params["startTime"] <= t <= params["endTime"]][:params["limit"]]
        if ep == "openInterestHist":
            return [{"symbol": "X", "sumOpenInterest": f"{1000 + (t - T0) // BAR_MS}", "sumOpenInterestValue": "5000",
                     "timestamp": t} for t in ts]
        if ep == "takerlongshortRatio":
            return [{"buySellRatio": "1.5", "sellVol": "1", "buyVol": "1.5", "timestamp": t - BAR_MS} for t in ts
                    if params["startTime"] <= t - BAR_MS <= params["endTime"]]
        r = {"topLongShortAccountRatio": "2.0", "topLongShortPositionRatio": "3.0",
             "globalLongShortAccountRatio": "4.0"}[ep]
        return [{"symbol": "X", "longAccount": "0.6", "longShortRatio": r, "shortAccount": "0.4", "timestamp": t}
                for t in ts]


def test_fetch_klines_only_closed_bars_and_pages():
    api = FakeAPI()
    df = live.fetch_klines("X", T0, get=api, now=NOW, pace=0)
    assert list(df.columns) == KLINE_COLS
    assert df["open_time"].iloc[0] == T0
    # bar opening at T0+100*BAR is still forming at NOW -> last closed is T0+99*BAR
    assert df["open_time"].iloc[-1] == T0 + 99 * BAR_MS and len(df) == 100
    assert df["open_time"].is_unique and df["trades"].iloc[0] == 7
    # more than one page
    big = live.fetch_klines("X", T0, get=api, now=T0 + 190 * BAR_MS, pace=0)
    assert len(big) == 189 or len(big) == 190


def test_klines_paging_beyond_1500(monkeypatch):
    calls = []

    def get(path, p):
        calls.append(p)
        t = p["startTime"]
        return [[t + i * BAR_MS, "1", "1", "1", "1", "1", 0, "1", 1, "0", "0", "0"]
                for i in range(1500) if t + i * BAR_MS <= p["endTime"]]
    now = T0 + 4000 * BAR_MS + 1
    df = live.fetch_klines("X", T0, get=get, now=now, pace=0)
    assert len(calls) >= 3 and len(df) == 4000 and df["open_time"].is_unique


def test_metric_timestamps_are_mapped_onto_archive_convention():
    api = FakeAPI()
    df = live.fetch_metrics("X", T0, get=api, now=NOW, pace=0)
    assert list(df.columns) == METRIC_COLS
    # REST stamps OI/ratios at period END (T0+BAR..), archive at period START: shifted by -5m.
    assert df["create_time"].iloc[0] == T0
    assert df["sum_open_interest"].iloc[0] == 1001.0
    assert df["count_toptrader_long_short_ratio"].iloc[0] == 2.0
    assert df["sum_toptrader_long_short_ratio"].iloc[0] == 3.0
    assert df["count_long_short_ratio"].iloc[0] == 4.0
    assert df["sum_taker_long_short_vol_ratio"].iloc[0] == 1.5     # taker is already bar-open stamped
    assert df["create_time"].iloc[-1] <= NOW // BAR_MS * BAR_MS - BAR_MS      # no partial periods
    assert df["create_time"].is_unique and df["create_time"].is_monotonic_increasing


def test_metrics_limited_to_29_days():
    api = FakeAPI()
    live.fetch_metrics("X", 0, get=api, now=NOW, pace=0)
    starts = [p["startTime"] for path, p in api.calls if "futures/data" in path]
    assert min(starts) >= NOW - 29 * 86_400_000 - BAR_MS


def test_funding_floors_to_seconds_and_merges():
    api = FakeAPI()
    f = live.fetch_funding("X", T0, get=api, pace=0)
    assert f["calc_time"].tolist() == [T0, T0 + 28_800_000, T0 + 2 * 28_800_000]
    assert f["funding_rate"].iloc[0] == 0.0001


def test_top_up_merges_into_parquet_and_is_idempotent(tmp_path):
    api = FakeAPI()
    # seed klines with the first 50 bars
    seed = live.fetch_klines("X", T0, T0 + 49 * BAR_MS, get=api, now=NOW, pace=0)
    frames.write_parquet_atomic(seed, tmp_path / "klines" / "5m" / "X.parquet")
    added = live.top_up("X", tmp_path, now=NOW, get=api)
    assert added["klines"] == 50 and added["funding"] == 3
    kl = pd.read_parquet(tmp_path / "klines" / "5m" / "X.parquet")
    assert len(kl) == 100 and kl["open_time"].is_unique and kl["open_time"].is_monotonic_increasing
    again = live.top_up("X", tmp_path, now=NOW, get=api)
    assert again["klines"] == 0 and again["metrics"] == 0 and again["funding"] == 0


def test_rest_get_raises_on_client_error(monkeypatch):
    class R:
        status_code = 400
        headers = {}
        text = "bad"

        def json(self):
            return {}
    monkeypatch.setattr(live._session, "get", lambda *a, **k: R())
    with pytest.raises(RuntimeError):
        live.rest_get("/x", {})
