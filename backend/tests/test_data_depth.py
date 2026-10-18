"""bookDepth: 30 s snapshots -> 5m rows (hand-made inputs), incremental download, loader/quality glue."""
import datetime as dt
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

from test_sim_helpers import BAR, mk_klines
from tb.data import archive, depth, frames, loader, quality
from tb.data.paths import depth_meta_path, depth_path, kline_path

DAY = dt.date(2026, 1, 1)
D0 = archive.to_ms(DAY)
BANDS_NEW = [-5, -4, -3, -2, -1, -0.2, 0.2, 1, 2, 3, 4, 5]
BANDS_OLD = [b for b in BANDS_NEW if abs(b) != 0.2]


def val(band, k):
    """Deterministic notional/depth for band (signed) at snapshot index k."""
    return (3.0 if band < 0 else 2.0) * abs(band) * 100.0 + k


def csv_text(snaps, bands=BANDS_NEW, fmt="{:.2f}", drop=()):
    """snaps: list of (offset_seconds_from_day_start, k)."""
    lines = ["timestamp,percentage,depth,notional"]
    for off, k in snaps:
        ts = (dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=off)).strftime("%Y-%m-%d %H:%M:%S")
        for b in bands:
            if (k, b) in drop:
                continue
            lines.append(f"{ts},{fmt.format(b)},{val(b, k) / 10:.8f},{val(b, k):.8f}")
    return "\n".join(lines) + "\n"


def zbytes(text):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x.csv", text)
    return b.getvalue()


def agg(snaps, **kw):
    return depth.aggregate_day(depth.parse_depth_csv(zbytes(csv_text(snaps, **kw))), D0)


def test_last_snapshot_per_bar_mean_and_boundaries():
    # bar 00:00: snapshots at 00:00:03 .. 00:04:33 (k=0..9); one stamped exactly 00:05:00 (k=10) -> same bar
    # (window is (open, close]); 00:05:03 (k=11) -> bar 00:05 ; nothing in 00:10 ; 00:15:03 (k=12) -> bar 00:15
    snaps = [(3 + 30 * k, k) for k in range(10)] + [(300, 10), (303, 11), (903, 12)]
    d = agg(snaps)
    assert d["open_time"].tolist() == [D0, D0 + BAR, D0 + 3 * BAR]          # empty 00:10 bar has no row
    assert d["snap_count"].tolist() == [11, 1, 1]
    r0 = d.iloc[0]
    assert r0["bid_notional_1"] == pytest.approx(val(-1, 10))                # LAST snapshot at/before close
    assert r0["ask_notional_5"] == pytest.approx(val(5, 10))
    assert r0["bid_depth_3"] == pytest.approx(val(-3, 10) / 10)
    assert r0["ask_depth_0.2"] == pytest.approx(val(0.2, 10) / 10, rel=1e-6)
    assert r0["bid_notional_1_mean"] == pytest.approx(np.mean([val(-1, k) for k in range(11)]), rel=1e-6)
    assert r0["ask_notional_1_mean"] == pytest.approx(np.mean([val(1, k) for k in range(11)]), rel=1e-6)
    assert d.iloc[1]["bid_notional_1"] == pytest.approx(val(-1, 11))
    assert d.iloc[1]["bid_notional_1_mean"] == pytest.approx(val(-1, 11))   # single snapshot: mean == last
    assert d.iloc[2]["ask_notional_2"] == pytest.approx(val(2, 12))


def test_no_lookahead_snapshot_after_close_is_not_in_bar():
    d = agg([(3, 0), (303, 1)])                      # 00:05:03 is after bar 00:00's close
    assert d.iloc[0]["bid_notional_1"] == pytest.approx(val(-1, 0)) and d.iloc[0]["snap_count"] == 1


def test_snapshot_at_day_start_is_clipped_into_first_bar():
    d = agg([(0, 0), (30, 1)])
    assert d["open_time"].tolist() == [D0] and d.iloc[0]["snap_count"] == 2
    assert d.iloc[0]["bid_notional_1"] == pytest.approx(val(-1, 1))


def test_old_files_without_02_band_and_integer_percentages():
    d = agg([(3, 0), (33, 1)], bands=BANDS_OLD, fmt="{:g}")      # '-5', '1' ... no +-0.2
    assert d["bid_notional_0.2"].isna().all() and d["ask_depth_0.2"].isna().all()
    assert d.iloc[0]["ask_notional_4"] == pytest.approx(val(4, 1))
    assert list(d.columns) == depth.DEPTH_COLS


def test_incomplete_snapshot_gives_nan_for_that_cell_only():
    d = agg([(3, 0), (33, 1)], drop={(1, 5)})                    # last snapshot lacks the +5 band
    assert np.isnan(d.iloc[0]["ask_notional_5"]) and np.isnan(d.iloc[0]["ask_depth_5"])
    assert d.iloc[0]["ask_notional_4"] == pytest.approx(val(4, 1))
    assert d.iloc[0]["bid_notional_5"] == pytest.approx(val(-5, 1))


def test_column_schema_and_dtypes():
    d = agg([(3, 0)])
    assert d.columns[0] == "open_time" and d["open_time"].dtype == "int64"
    assert d["snap_count"].dtype == "int16" and d["bid_notional_1"].dtype == "float32"
    for s in ("bid", "ask"):
        for b in ("0.2", "1", "2", "3", "4", "5"):
            assert f"{s}_depth_{b}" in d.columns and f"{s}_notional_{b}" in d.columns
    assert "bid_notional_1_mean" in d.columns and "ask_notional_1_mean" in d.columns
    assert len(depth.empty_depth()) == 0 and list(depth.empty_depth().columns) == depth.DEPTH_COLS


def test_bad_zip_member_and_columns_rejected():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x.txt", "nope")
    with pytest.raises(ValueError):
        depth.parse_depth_csv(b.getvalue())
    with pytest.raises(ValueError):
        depth.parse_depth_csv(zbytes("2026-01-01 00:00:03,-5,1,2\n"))


# --------------------------------------------------------------------------- download
class FakeDepthHTTP:
    def __init__(self, missing=("2026-01-03",)):
        self.requests = []
        self.missing = set(missing)

    def __call__(self, url, retries=5, timeout=60.0):
        self.requests.append(url)
        day = url.rsplit("-bookDepth-", 1)[1][:10]
        if day in self.missing or "bookDepth" not in url:
            return None
        d = dt.date.fromisoformat(day)
        off = (d - DAY).days
        text = csv_text([(3 + 30 * k, k + off * 100) for k in range(20)])
        text = text.replace("2026-01-01", day)                 # csv_text stamps 2026-01-01; move to `day`
        return zbytes(text)


def test_download_depth_incremental_missing_days_and_checkpoint(monkeypatch, tmp_path):
    f = FakeDepthHTTP()
    monkeypatch.setattr(archive, "http_get", f)
    kl = mk_klines(np.full(10, 100.0), t0=D0)                      # first kline on 2026-01-01
    frames.write_parquet_atomic(frames.normalize_klines(kl), kline_path("XUSDT", tmp_path))
    monkeypatch.setattr(depth, "CHECKPOINT_DAYS", 2)
    today = dt.date(2026, 1, 6)                                    # window: days 01-01 .. 01-05
    s = depth.download_depth("XUSDT", 1, tmp_path, today=today)
    assert s["downloaded"] == 4 and s["missing"] == 1 and s["failed"] == 0
    dp = pd.read_parquet(depth_path("XUSDT", tmp_path))
    assert dp["open_time"].is_unique and dp["open_time"].is_monotonic_increasing
    assert len(dp) == 4 * 2                                       # 20 snapshots (10 min) -> 2 bars/day
    meta = json.loads(depth_meta_path("XUSDT", tmp_path).read_text())
    assert meta["missing_days"] == ["2026-01-03"] and meta["rows"] == 8
    assert meta["first_day"] == "2026-01-01" and meta["last_day"] == "2026-01-05" and meta["has_02_band"]
    n = len(f.requests)
    s2 = depth.download_depth("XUSDT", 1, tmp_path, today=today)   # nothing new: no requests, 404 not retried
    assert len(f.requests) == n and s2["downloaded"] == 0 and s2["skipped"] == 5
    s3 = depth.download_depth("XUSDT", 1, tmp_path, today=dt.date(2026, 1, 7))     # one new day
    assert s3["downloaded"] == 1 and s3["rows_added"] == 2


def test_depth_job_is_registered_and_listed(tmp_path):
    assert "depth" in archive.KINDS and archive._job("depth") is depth.download_depth
    frames.write_parquet_atomic(depth.empty_depth(), depth_path("QQQUSDT", tmp_path))
    assert loader.symbols_available(tmp_path, "depth") == ["QQQUSDT"]


# --------------------------------------------------------------------------- loader / quality
def _depth_for_klines(n_bars, skip=()):
    rows = []
    for i in range(n_bars):
        if i in skip:
            continue
        rows.append(agg([(3 + 300 * i, i)]).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True).astype(depth.empty_depth().dtypes.to_dict())


def test_align_depth_exact_match_and_ffill():
    kl = mk_klines(np.full(8, 100.0), t0=D0)
    dp = _depth_for_klines(8, skip=(3, 4))
    al = loader.align_depth(kl, dp)
    assert len(al) == 8 and al["has_depth"].tolist() == [1, 1, 1, 0, 0, 1, 1, 1]
    assert np.isnan(al["bid_notional_1"].iloc[3]) and al["bid_notional_1"].iloc[5] == pytest.approx(val(-1, 5))
    ff = loader.align_depth(kl, dp, max_ffill_bars=1)
    assert ff["bid_notional_1"].iloc[3] == pytest.approx(val(-1, 2)) and np.isnan(ff["bid_notional_1"].iloc[4])
    assert ff["has_depth"].iloc[3] == 0.0                      # flag stays honest
    e = loader.align_depth(kl, depth.empty_depth())
    assert e["has_depth"].sum() == 0 and len(e) == 8


def test_quality_report_covers_depth(tmp_path):
    n = 120
    kl = mk_klines(np.full(n, 100.0), t0=D0)
    frames.write_parquet_atomic(frames.normalize_klines(kl), kline_path("DDDUSDT", tmp_path))
    dp = _depth_for_klines(n, skip=range(40, 60))                # 20-bar hole
    dp.loc[5, "snap_count"] = 1
    frames.write_parquet_atomic(dp, depth_path("DDDUSDT", tmp_path))
    rep = quality.quality_report("DDDUSDT", tmp_path, now_ms=D0 + 300 * BAR)
    assert rep["depth_rows"] == n - 20 and rep["depth_gaps_over_1h"] == 1
    assert rep["depth_coverage"] == pytest.approx((n - 20) / n)
    assert rep["depth_low_snap_bars"] >= 1 and rep["depth_nonpositive_rows"] == 0
    assert rep["depth_nonmonotone_rows"] == 0 and rep["depth_unsorted_or_duplicate"] == 0
    # symbol without depth: report still works
    frames.write_parquet_atomic(frames.normalize_klines(kl), kline_path("NODEPTHUSDT", tmp_path))
    assert quality.quality_report("NODEPTHUSDT", tmp_path)["depth_rows"] == 0
