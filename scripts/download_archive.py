#!/usr/bin/env python
"""Download / refresh the Binance USDⓈ-M archive (klines 5m, metrics, funding) into data/archive/.

  .venv/bin/python scripts/download_archive.py                       # default 25 symbols, 12 months, all kinds
  .venv/bin/python scripts/download_archive.py --symbols BTCUSDT,ETHUSDT --months 3 --kinds klines
  .venv/bin/python scripts/download_archive.py --no-topup            # skip the REST top-up to "now"

Incremental: months/days already in the parquet files are skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from tb.data import archive, live  # noqa: E402
from tb.data.paths import archive_root  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=",".join(archive.DEFAULT_SYMBOLS),
                    help="comma list (default: %(default)s)")
    ap.add_argument("--all-perps", action="store_true",
                    help="every TRADING USDT perpetual listed >= --min-age-months ago (overrides --symbols)")
    ap.add_argument("--min-age-months", type=int, default=24)
    ap.add_argument("--months", type=int, default=12, help="full months before the current one")
    ap.add_argument("--kinds", default="klines,metrics,funding")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--root", default=None, help="archive dir (default data/archive or $TB_ARCHIVE_DIR)")
    ap.add_argument("--no-verify", action="store_true", help="skip .CHECKSUM verification")
    ap.add_argument("--no-topup", action="store_true", help="skip REST top-up after the archive pass")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)

    if a.all_perps:
        symbols = archive.universe_symbols(a.min_age_months)
        archive.log(f"universe: {len(symbols)} USDT perps listed >= {a.min_age_months} months ago")
    else:
        symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    bad = [k for k in kinds if k not in archive.KINDS]
    if bad:
        ap.error(f"unknown kinds {bad}")
    root = Path(a.root) if a.root else archive_root()

    if not a.summary_only:
        res = archive.download_all(symbols, kinds, a.months, a.workers, root, not a.no_verify)
        errs = {k: v for k, v in res.items() if isinstance(v, str) or v.get("failed")}
        if errs:
            archive.log(f"WARNING: {len(errs)} jobs had failures (re-run to retry): "
                        f"{sorted(errs)[:10]}")
        if not a.no_topup:
            archive.log("REST top-up ...")
            for i, s in enumerate(symbols, 1):
                try:
                    added = live.top_up(s, root, kinds=tuple(kinds))
                    archive.log(f"  topup [{i}/{len(symbols)}] {s}: {added}")
                except Exception as e:  # noqa: BLE001
                    archive.log(f"  topup {s} FAILED: {type(e).__name__}: {e}")
    summ = archive.summarize(symbols, root)
    with __import__("pandas").option_context("display.max_rows", 500, "display.width", 200):
        print(summ.to_string(index=False))
    print("total MB:", round(float(summ.get("mb", 0).sum()), 1) if "mb" in summ else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
