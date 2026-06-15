#!/usr/bin/env python3
"""
5-YEAR HISTORICAL DATA DOWNLOADER (15-min bars)
================================================
Downloads 5 years of 15-minute OHLCV data for SOL, ETH, LINK, AVAX from
Coinbase via CCXT (public endpoint — no API key needed).

Saves to data/historical_5yr/ so it does NOT overwrite your existing
data/historical/ folder. This way you can keep both side-by-side.

Resume support: if download is interrupted, re-running picks up where it
left off (skips coins already complete; appends to partial files).

Volume:
    5 years × 365 days × 96 bars/day = ~175,000 candles per coin
    × 4 coins = ~700,000 candles total
    Coinbase rate limit ~0.5 sec/request, 300 candles/request →
    ~10-20 minutes total. Network-bound, not CPU-bound.

Usage:
    python3.10 download_data_5yr.py                          # All 4 coins
    python3.10 download_data_5yr.py --symbols SOL ETH        # subset
    python3.10 download_data_5yr.py --start-date 2021-06-01  # custom start
    python3.10 download_data_5yr.py --force                  # re-download all
"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import ccxt
except ImportError:
    print("ERROR: ccxt not installed. Run: pip3.10 install ccxt")
    sys.exit(1)

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "historical_5yr"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Match the coins the HMM classifier covers
DEFAULT_SYMBOLS = ["SOL", "ETH", "LINK", "AVAX"]
DEFAULT_DAYS = 5 * 365 + 1   # 1826
TIMEFRAME = "15m"

# Timeframe in milliseconds (15m = 900,000 ms)
TF_MS = 900_000
# Coinbase Advanced Trade public candles endpoint: 300 candles max per request
CANDLES_PER_REQUEST = 300


def daterange_chunks(start_ms: int, end_ms: int, step_ms: int):
    """Yield (since, until) ms pairs walking forward in chunks."""
    cur = start_ms
    while cur < end_ms:
        yield cur, min(cur + step_ms, end_ms)
        cur += step_ms


def existing_endpoint(path: Path) -> int | None:
    """Return last timestamp (ms) in an existing CSV, or None."""
    if not path.exists() or path.stat().st_size < 100:
        return None
    try:
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        if df.empty:
            return None
        last_ts = df.index[-1]
        return int(last_ts.timestamp() * 1000)
    except Exception:
        return None


def download_coin(exchange, symbol: str, start_ms: int, end_ms: int,
                    out_path: Path, resume_from: int | None,
                    verbose: bool = True) -> int:
    """Download candles for one coin into out_path. Returns count saved."""
    if resume_from is not None:
        start_ms = max(start_ms, resume_from + TF_MS)
        if start_ms >= end_ms:
            if verbose:
                print(f"    Already complete (last bar: "
                      f"{pd.Timestamp(resume_from, unit='ms')})")
            return 0
        if verbose:
            print(f"    Resuming from {pd.Timestamp(start_ms, unit='ms')}")

    step_ms = TF_MS * CANDLES_PER_REQUEST
    total_candles = 0
    write_mode = "a" if (resume_from is not None and out_path.exists()) else "w"
    write_header = (write_mode == "w")

    try:
        for chunk_since, chunk_until in daterange_chunks(start_ms, end_ms, step_ms):
            attempts = 0
            while attempts < 5:
                try:
                    candles = exchange.fetch_ohlcv(
                        symbol, timeframe=TIMEFRAME,
                        since=chunk_since,
                        limit=CANDLES_PER_REQUEST,
                    )
                    break
                except ccxt.RateLimitExceeded:
                    time.sleep(5)
                    attempts += 1
                except ccxt.NetworkError as e:
                    if verbose:
                        print(f"\n    Network error, retrying: {e}")
                    time.sleep(3)
                    attempts += 1
                except Exception as e:
                    if verbose:
                        print(f"\n    Error: {e}")
                    time.sleep(2)
                    attempts += 1
            else:
                if verbose:
                    print(f"\n    Giving up after 5 retries at "
                          f"{pd.Timestamp(chunk_since, unit='ms')}")
                break

            if not candles:
                # Either reached end-of-history or product wasn't trading yet
                # Step forward and try again
                continue

            # Filter to expected window (some exchanges return overshoot)
            candles = [c for c in candles if start_ms <= c[0] < end_ms]
            if not candles:
                continue

            df = pd.DataFrame(candles, columns=["timestamp", "open", "high",
                                                  "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df[~df.index.duplicated(keep="first")]

            df.to_csv(out_path, mode=write_mode, header=write_header,
                        index_label="timestamp")
            write_mode = "a"
            write_header = False

            total_candles += len(df)

            # Progress
            last_ts = df.index[-1]
            total_days_done = (last_ts - pd.Timestamp(start_ms, unit="ms")).days
            if verbose:
                pct = (chunk_until - start_ms) / (end_ms - start_ms) * 100
                print(f"    {total_candles:>7,} candles  "
                      f"({total_days_done:>4}d done, {pct:.0f}%)  "
                      f"last={last_ts}", end="\r")

            time.sleep(exchange.rateLimit / 1000)

    except KeyboardInterrupt:
        print(f"\n    Interrupted. Saved {total_candles} candles. "
              f"Re-run to resume.")
        raise

    if verbose:
        print(f"    {total_candles:>7,} candles total" + " " * 40)

    # Final dedup + sort
    if out_path.exists():
        df_all = pd.read_csv(out_path, index_col="timestamp", parse_dates=True)
        df_all = df_all[~df_all.index.duplicated(keep="first")]
        df_all.sort_index(inplace=True)
        df_all.to_csv(out_path, index_label="timestamp")
        return len(df_all)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Download 5 years of 15m crypto data from Coinbase")
    ap.add_argument("--symbols", "-s", nargs="+", default=DEFAULT_SYMBOLS,
                    help=f"Coins (default: {DEFAULT_SYMBOLS})")
    ap.add_argument("--days", "-d", type=int, default=DEFAULT_DAYS,
                    help=f"Days back from today (default: {DEFAULT_DAYS} = 5y)")
    ap.add_argument("--start-date", type=str, default=None,
                    help="Override start date (YYYY-MM-DD)")
    ap.add_argument("--force", action="store_true",
                    help="Re-download from scratch (deletes existing files)")
    args = ap.parse_args()

    now_utc = datetime.now(timezone.utc)
    if args.start_date:
        start_dt = datetime.fromisoformat(args.start_date).replace(
            tzinfo=timezone.utc)
    else:
        start_dt = now_utc - timedelta(days=args.days)
    end_dt = now_utc
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print("=" * 70)
    print("  5-YEAR HISTORICAL DATA DOWNLOADER")
    print("=" * 70)
    print(f"  Source:     Coinbase Advanced Trade (public, no auth)")
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Timeframe:  {TIMEFRAME}")
    print(f"  Period:     {start_dt.date()} → {end_dt.date()}  "
          f"({(end_dt - start_dt).days} days)")
    print(f"  Output:     {DATA_DIR}")
    print(f"  Force:      {args.force}")
    print("=" * 70)

    exchange = ccxt.coinbase({"enableRateLimit": True})

    summary = []
    for coin in args.symbols:
        symbol = f"{coin}/USD"
        out_path = DATA_DIR / f"{coin}_USD_15m.csv"

        print(f"\n  ── {symbol} ──")

        if args.force and out_path.exists():
            out_path.unlink()
            print(f"    [--force] Deleted existing file")

        resume_from = None if args.force else existing_endpoint(out_path)
        try:
            n = download_coin(exchange, symbol, start_ms, end_ms,
                                out_path, resume_from)
        except KeyboardInterrupt:
            print(f"\n  Interrupted. Re-run to resume.")
            sys.exit(1)

        if out_path.exists():
            df = pd.read_csv(out_path, index_col="timestamp",
                              parse_dates=True)
            size_mb = out_path.stat().st_size / 1024 / 1024
            summary.append({
                "coin": coin,
                "rows": len(df),
                "first": df.index[0],
                "last": df.index[-1],
                "size_mb": size_mb,
                "path": out_path,
            })
            print(f"    Saved: {out_path.name} ({size_mb:.1f} MB, "
                  f"{len(df):,} rows)")
            print(f"    Range: {df.index[0]} → {df.index[-1]}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"  {'Coin':<6} {'Rows':>9}  {'First':<20} {'Last':<20}  Size")
    print("  " + "─" * 64)
    for s in summary:
        print(f"  {s['coin']:<6} {s['rows']:>9,}  "
              f"{str(s['first']):<20} {str(s['last']):<20}  "
              f"{s['size_mb']:.1f} MB")

    print()
    print(f"  All data saved to: {DATA_DIR}")
    print()
    print(f"  Next step — re-run the regime comparison on 5-year data:")
    print(f"    python3.10 backtest_with_regime.py \\")
    print(f"        --data-dir data/historical_5yr \\")
    print(f"        --labels-dir /Users/blakeweiss33/Desktop/regime_classifier/data/hmm_labels")


if __name__ == "__main__":
    main()
