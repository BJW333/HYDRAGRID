#!/usr/bin/env python3
"""
Test Alpaca Connection
======================
Quick check that your Alpaca paper trading keys work.

Usage:
    export ALPACA_API_KEY="your-key"
    export ALPACA_API_SECRET="your-secret"
    python test_alpaca.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.alpaca_adapter import AlpacaExchange


def main():
    key = os.environ.get('ALPACA_API_KEY', '')
    secret = os.environ.get('ALPACA_API_SECRET', '')

    if not key or not secret:
        print("❌ Set ALPACA_API_KEY and ALPACA_API_SECRET env vars first")
        print("   export ALPACA_API_KEY='your-key'")
        print("   export ALPACA_API_SECRET='your-secret'")
        sys.exit(1)

    print("🔌 Connecting to Alpaca paper trading...")
    exchange = AlpacaExchange({
        'apiKey': key,
        'secret': secret,
        'paper': True,
    })

    # 1. Account info
    print("\n📊 Account:")
    balance = exchange.fetch_balance()
    print(f"   Cash:   ${balance['USD']['free']:,.2f}")
    print(f"   Equity: ${balance['USD']['total']:,.2f}")

    # 2. Fetch crypto bars
    print("\n📈 Fetching BTC/USD 4h candles...")
    candles = exchange.fetch_ohlcv('BTC/USD', '4h', limit=5)
    if candles:
        for c in candles[-3:]:
            from datetime import datetime
            ts = datetime.utcfromtimestamp(c[0] / 1000).strftime('%Y-%m-%d %H:%M')
            print(f"   {ts}  O:{c[1]:,.0f}  H:{c[2]:,.0f}  L:{c[3]:,.0f}  C:{c[4]:,.0f}  V:{c[5]:,.2f}")
    else:
        print("   ⚠️  No candles returned")

    # 3. Ticker
    print("\n💲 Current prices:")
    for sym in ['BTC/USD', 'ETH/USD', 'LINK/USD', 'SOL/USD']:
        try:
            ticker = exchange.fetch_ticker(sym)
            print(f"   {sym}: ${ticker['last']:,.2f}")
        except Exception as e:
            print(f"   {sym}: ❌ {e}")

    print("\n✅ Alpaca paper trading is ready!")
    print("   Run: python live_trader.py")


if __name__ == '__main__':
    main()
