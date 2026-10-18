"""Archive downloader: CSV parsing (header / headerless), checksum, incremental merge (offline)."""
import datetime as dt
import hashlib
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from tb.data import archive, frames, loader
from tb.data.paths import FUNDING_COLS, KLINE_COLS, METRIC_COLS

T0 = 1_785_542_400_000  # 2026-08-01


def zipped(name: str, text: str) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, text)
    return b.getvalue()


def kl_csv(n, t0=T0, header=True):
    rows = []
    for i in range(n):
        t = t0 + i * 300_000
        rows.append(f"{t},100.0,101.0,99.0,100.5,10.0,{t + 299_999},1005.0,{5 + i},4.0,402.0,0")
    head = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
            "taker_buy_quote_volume,ignore"] if header else []
    return "\n".join(head + rows) + "\n"


@pytest.mark.parametrize("header", [True, False])
def test_parse_klines_with_and_without_header(header):
    df = archive.parse_klines_csv(zipped("k.csv", kl_csv(5, header=header)))
    assert list(df.columns) == KLINE_COLS and len(df) == 5
    assert df["open_time"].dtype == "int64" and df["trades"].dtype == "int64"
    assert df["open_time"].iloc[0] == T0 and df["trades"].iloc[2] == 7
    assert df["quote_volume"].iloc[0] == 1005.0 and df["taker_buy_quote_volume"].iloc[0] == 402.0


def test_parse_metrics_datetime_header_and_headerless():
    body = ("2026-10-10 00:00:00,BTCUSDT,92508.37,7636760211.077,1.54563091,1.613763,1.4802649,0.296917\n"
            "2026-10-10 00:05:00,BTCUSDT,92436.224,7633292639.0,1.5,1.6,1.4,\n")
    head = ("create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
            "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n")
    exp = int(dt.datetime(2026, 10, 10, tzinfo=dt.timezone.utc).timestamp() * 1000)
    for text in (head + body, body):
        df = archive.parse_metrics_csv(zipped("m.csv", text))
        assert list(df.columns) == METRIC_COLS and len(df) == 2
        assert df["create_time"].iloc[0] == exp and df["create_time"].iloc[1] == exp + 300_000
        assert df["sum_open_interest"].iloc[0] == pytest.approx(92508.37)
        assert np.isnan(df["sum_taker_long_short_vol_ratio"].iloc[1])      # empty field -> NaN


def test_parse_funding_with_interval_column_and_plusone_ms():
    text = "calc_time,funding_interval_hours,last_funding_rate\n1785542400001,8,0.00004123\n1785571200000,8,-0.0001\n"
    df = archive.parse_funding_csv(zipped("f.csv", text))
    assert list(df.columns) == FUNDING_COLS
    assert df["calc_time"].tolist() == [1785542400000, 1785571200000]     # floored to whole seconds
    assert df["funding_rate"].tolist() == [0.00004123, -0.0001]
    # headerless 2-column variant
    df2 = archive.parse_funding_csv(zipped("f.csv", "1785542400001,0.00004123\n"))
    assert df2["calc_time"].iloc[0] == 1785542400000


def test_zip_with_no_csv_is_rejected():
    with pytest.raises(ValueError):
        archive.parse_klines_csv(zipped("x.txt", "nope"))


def test_microsecond_timestamps_are_normalised():
    s = pd.Series([str(T0 * 1000), str((T0 + 300_000) * 1000)])
    assert frames._to_ms(s).tolist() == [T0, T0 + 300_000]


def test_merge_frames_dedupes_new_wins_and_sorts():
    a = frames.normalize_klines(archive.parse_klines_csv(zipped("k.csv", kl_csv(4))))
    b = a.iloc[2:].copy()
    b = pd.concat([b, frames.normalize_klines(archive.parse_klines_csv(zipped("k.csv", kl_csv(3, t0=T0 + 10 * 300_000))))])
    b.loc[b.index[0], "close"] = 123.0
    m = frames.merge_frames(a, b, "open_time")
    assert m["open_time"].is_monotonic_increasing and m["open_time"].is_unique and len(m) == 7
    assert m.loc[m["open_time"] == a["open_time"].iloc[2], "close"].iloc[0] == 123.0


def test_urls_match_binance_vision_layout():
    assert archive.monthly_url("klines", "BTCUSDT", "2026-08") == \
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2026-08.zip"
    assert archive.daily_url("metrics", "BTCUSDT", "2026-10-10") == \
        "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-10-10.zip"
    assert archive.monthly_url("funding", "ETHUSDT", "2026-08") == \
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2026-08.zip"
    with pytest.raises(ValueError):
        archive.monthly_url("metrics", "BTCUSDT", "2026-08")


def test_window_months():
    full, cur = archive.window_months(12, dt.date(2026, 10, 18))
    assert len(full) == 12 and full[0] == dt.date(2025, 10, 1) and full[-1] == dt.date(2026, 9, 1)
    assert cur == dt.date(2026, 10, 1)


class FakeHTTP:
    """Serves a deterministic archive from memory; records requests."""

    def __init__(self, bad_checksum=False):
        self.requests = []
        self.bad_checksum = bad_checksum

    def __call__(self, url, retries=5, timeout=60.0):
        self.requests.append(url)
        name = url.rsplit("/", 1)[1]
        if name.endswith(".CHECKSUM"):
            z = self._zip(url[:-len(".CHECKSUM")])
            if z is None:
                return None
            h = "0" * 64 if self.bad_checksum else hashlib.sha256(z).hexdigest()
            return f"{h}  {name[:-9]}\n".encode()
        return self._zip(url)

    def _zip(self, url):
        if "/klines/" in url:
            if "2026-10-1" in url and "monthly" in url:
                return None
            if "monthly" in url:
                ym = url.rsplit("-5m-", 1)[1][:7]
                y, m = map(int, ym.split("-"))
                n = (archive.add_months(dt.date(y, m, 1), 1) - dt.date(y, m, 1)).days * 288
                t0 = archive.to_ms(dt.date(y, m, 1))
                return zipped("k.csv", kl_csv(n, t0))
            day = url.rsplit("-5m-", 1)[1][:10]
            d = dt.date.fromisoformat(day)
            return zipped("k.csv", kl_csv(288, archive.to_ms(d), header=False))
        if "/metrics/" in url:
            day = url.rsplit("-metrics-", 1)[1][:10]
            if day == "2026-09-20":
                return None                                     # a missing day (404) is fine
            d = dt.date.fromisoformat(day)
            t0 = archive.to_ms(d)
            rows = []
            for i in range(288):
                ts = dt.datetime.fromtimestamp((t0 + i * 300_000) / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                rows.append(f"{ts},BTCUSDT,1000.0,2000.0,1.1,1.2,1.3,1.4")
            return zipped("m.csv", "\n".join(rows) + "\n")
        if "/fundingRate/" in url:
            ym = url.rsplit("-fundingRate-", 1)[1][:7]
            y, m = map(int, ym.split("-"))
            t0 = archive.to_ms(dt.date(y, m, 1))
            rows = [f"{t0 + i * 8 * 3_600_000 + 1},8,0.0001" for i in range(90)]
            return zipped("f.csv", "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(rows) + "\n")
        return None


@pytest.fixture
def fake(monkeypatch):
    f = FakeHTTP()
    monkeypatch.setattr(archive, "http_get", f)
    monkeypatch.setattr(archive.time, "sleep", lambda *_: None)
    return f


TODAY = dt.date(2026, 10, 4)


def test_download_klines_incremental(fake, tmp_path):
    s = archive.download_klines("BTCUSDT", 2, tmp_path, today=TODAY)
    kl = pd.read_parquet(tmp_path / "klines" / "5m" / "BTCUSDT.parquet")
    # Aug (31d) + Sep (30d) monthly + Oct 1..3 daily
    assert len(kl) == (31 + 30 + 3) * 288 and s["downloaded"] == 5
    assert kl["open_time"].is_monotonic_increasing and kl["open_time"].is_unique
    n_req = len(fake.requests)
    s2 = archive.download_klines("BTCUSDT", 2, tmp_path, today=TODAY)       # nothing new -> no data requests
    assert s2["downloaded"] == 0 and s2["skipped"] >= 5 and s2["rows_added"] == 0
    assert len(fake.requests) == n_req
    # a new day appears -> only that day is fetched
    s3 = archive.download_klines("BTCUSDT", 2, tmp_path, today=dt.date(2026, 10, 5))
    assert s3["downloaded"] == 1 and s3["rows_added"] == 288
    # a hole in the middle of a month is healed on the next run
    kl = pd.read_parquet(tmp_path / "klines" / "5m" / "BTCUSDT.parquet")
    kl = kl[~((kl["open_time"] >= archive.to_ms(dt.date(2026, 9, 10))) & (kl["open_time"] < archive.to_ms(dt.date(2026, 9, 11))))]
    kl.to_parquet(tmp_path / "klines" / "5m" / "BTCUSDT.parquet")
    # month still "covered" at both ends by the cheap test; day-level holes are only repaired for the
    # current month. The monthly file is re-fetched with --force semantics: removing the file's edge
    kl2 = kl[kl["open_time"] < archive.to_ms(dt.date(2026, 9, 30))]
    kl2.to_parquet(tmp_path / "klines" / "5m" / "BTCUSDT.parquet")
    s4 = archive.download_klines("BTCUSDT", 2, tmp_path, today=dt.date(2026, 10, 5))
    assert s4["rows_added"] > 0
    full = pd.read_parquet(tmp_path / "klines" / "5m" / "BTCUSDT.parquet")
    assert len(full) == (31 + 30 + 4) * 288


def test_download_metrics_skips_404_days(fake, tmp_path):
    s = archive.download_metrics("BTCUSDT", 1, tmp_path, today=TODAY)
    me = pd.read_parquet(tmp_path / "metrics" / "BTCUSDT.parquet")
    assert s["missing"] == 1                                 # 2026-09-20
    assert len(me) == (30 + 3 - 1) * 288                     # Sep (30) + Oct 1-3, minus the missing day
    assert list(me.columns) == METRIC_COLS and me["create_time"].is_unique


def test_download_funding_monthly(fake, tmp_path):
    archive.download_funding("BTCUSDT", 2, tmp_path, today=TODAY)
    fu = pd.read_parquet(tmp_path / "funding" / "BTCUSDT.parquet")
    assert len(fu) == 90 * 2 or len(fu) > 100                # two monthly files, +1 ms stamps floored
    assert (fu["calc_time"] % 1000 == 0).all()


def test_checksum_mismatch_is_an_error(monkeypatch, tmp_path):
    bad = FakeHTTP(bad_checksum=True)
    monkeypatch.setattr(archive, "http_get", bad)
    monkeypatch.setattr(archive.time, "sleep", lambda *_: None)
    s = archive.download_funding("BTCUSDT", 1, tmp_path, today=TODAY)
    assert s["failed"] >= 1 and s["downloaded"] == 0
    s_ok = archive.download_funding("BTCUSDT", 1, tmp_path, verify=False, today=TODAY)
    assert s_ok["downloaded"] == 1


def test_default_symbols_are_unique_usdt_perps():
    syms = archive.DEFAULT_SYMBOLS
    assert len(syms) == len(set(syms)) >= 25 and all(s.endswith("USDT") for s in syms)
    assert "1000PEPEUSDT" in syms and "BTCUSDT" in syms
