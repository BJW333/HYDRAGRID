#!/usr/bin/env python3
"""
Historical Data Downloader
============================
Downloads 5m and 15m crypto data from Binance (free, no API key needed).
Saves to CSV for backtesting.

Usage:
    python3.10 download_data.py                         # Default: 365 days, SOL+ETH+BTC
    python3.10 download_data.py --days 365              # Full year
    python3.10 download_data.py --symbols SOL ETH BTC   # Specific coins
    python3.10 download_data.py --timeframe 5m          # 5-minute only
    python3.10 download_data.py --timeframe 15m         # 15-minute only
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_ohlcv(symbol: str, timeframe: str = '5m', days: int = 365) -> pd.DataFrame:
    """
    Download OHLCV data from Binance.
    No API key needed — public data.
    
    Args:
        symbol: e.g. 'SOL/USDT', 'ETH/USDT', 'BTC/USDT'
        timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
        days: Number of days of history
    
    Returns:
        DataFrame with timestamp index and OHLCV columns
    """
    exchange = ccxt.coinbase({'enableRateLimit': True})
    
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    end = int(datetime.now().timestamp() * 1000)
    
    all_candles = []
    current = since
    limit = 1000  # Binance max per request
    
    # Timeframe in ms
    tf_ms = {
        '1m': 60000, '5m': 300000, '15m': 900000,
        '30m': 1800000, '1h': 3600000, '4h': 14400000, '1d': 86400000,
    }
    step = tf_ms.get(timeframe, 300000) * limit
    
    print(f"  Downloading {symbol} {timeframe} ({days} days)...")
    
    while current < end:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe,
                since=current, limit=limit
            )
            
            if not candles:
                break
            
            all_candles.extend(candles)
            current = candles[-1][0] + 1
            
            progress_days = (current - since) / 86400000
            print(f"    {len(all_candles)} candles ({progress_days:.0f}/{days} days)...", end='\r')
            
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"\n    Error at {datetime.utcfromtimestamp(current/1000)}: {e}")
            time.sleep(2)
            continue
    
    print(f"    {len(all_candles)} candles total" + " " * 30)
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    
    return df


def save_data(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save DataFrame to CSV."""
    clean_symbol = symbol.replace('/', '_')
    filename = f"{clean_symbol}_{timeframe}.csv"
    filepath = DATA_DIR / filename
    df.to_csv(filepath)
    size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"    Saved: {filepath} ({size_mb:.1f} MB, {len(df)} rows)")
    return filepath


def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load data from CSV if it exists."""
    clean_symbol = symbol.replace('/', '_')
    filename = f"{clean_symbol}_{timeframe}.csv"
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    return df


def main():
    parser = argparse.ArgumentParser(description='Download historical crypto data')
    parser.add_argument('--symbols', '-s', nargs='+', default=['SOL', 'ETH', 'BTC'],
                        help='Coins to download (default: SOL ETH BTC)')
    parser.add_argument('--days', '-d', type=int, default=365,
                        help='Days of history (default: 365)')
    parser.add_argument('--timeframe', '-t', nargs='+', default=['5m', '15m'],
                        help='Timeframes (default: 5m 15m)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    
    print("HISTORICAL DATA DOWNLOADER")
    _ex = ccxt.coinbase()
    print(f"Source: {_ex.name}")
    print(f"Coins: {', '.join(args.symbols)}")
    print(f"Timeframes: {', '.join(args.timeframe)}")
    print(f"Period: {args.days} days")
    print("=" * 60)
    
    for coin in args.symbols:
        symbol = f"{coin}/USD"
        for tf in args.timeframe:
            df = download_ohlcv(symbol, tf, args.days)
            if len(df) > 0:
                save_data(df, symbol, tf)
                
                # Show summary
                print(f"    Range: {df.index[0]} → {df.index[-1]}")
                print(f"    Days: {(df.index[-1] - df.index[0]).days}")
                print()
    
    print("=" * 60)
    print("Done! Files saved in data/historical/")
    print("Run backtest with: python3.10 backtest.py --from-file")
    print("=" * 60)


if __name__ == '__main__':
    main()
