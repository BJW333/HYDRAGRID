# Consolidation Breakout Trading Bot

Algorithmic trading bot that detects consolidation patterns (Bollinger Band squeezes + tight price ranges) and trades breakouts with trend alignment.

## Performance (Backtested)

| Asset | Monthly Return | Win Rate | Profit Factor |
|-------|----------------|----------|---------------|
| LINK/USD | 3.97% | 91.7% | 7.67 |
| SOL/USD | 1.80% | 85.7% | 4.36 |
| XRP/USD | 1.17% | 75.0% | 2.19 |
| BTC/USD | 0.45% | 80.0% | 2.05 |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Backtest
```bash
# Basic backtest
python backtest.py --symbol LINK/USD

# Auto-tune parameters for any asset
python backtest.py --symbol SOL/USD --auto-tune
```

### 3. Paper Trading (Test Live Signals)
```bash
# Monitor LINK, SOL, XRP with paper trading
python live_trader.py

# Specific assets
python live_trader.py --assets LINK/USD SOL/USD

# Fast check interval (for testing)
python live_trader.py --interval 60
```

### 4. Live Trading (Real Money)
```bash
# Set API keys
export KRAKEN_API_KEY="your-key"
export KRAKEN_API_SECRET="your-secret"

# Run live
python live_trader.py --live
```

## Telegram Alerts Setup

1. Message @BotFather on Telegram
2. Send `/newbot` and follow instructions
3. Copy the token
4. Message your bot, then visit: `https://api.telegram.org/bot<TOKEN>/getUpdates`
5. Find your `chat_id` in the response
6. Set environment variables:
```bash
export TELEGRAM_BOT_TOKEN="your-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

## Strategy Logic

**Entry Conditions (ALL required):**
- Bollinger Band width in lowest 25% (squeeze)
- Price range < threshold over 20 candles (consolidation)
- Breakout candle breaks above/below range
- Volume > threshold × average
- Trend aligned (MA slope + price position)

**Exit Rules:**
- Stop Loss: 2.0 × ATR from entry
- Profit Target: 2.2 × ATR from entry

## Files

```
breakout_bot/
├── backtest.py          # Backtesting CLI
├── live_trader.py       # Live/paper trading bot
├── config/
│   └── settings.py      # All parameters
├── src/
│   ├── data_fetcher.py  # Exchange data
│   ├── indicators.py    # Technical indicators
│   ├── signals.py       # Signal generation
│   ├── backtester.py    # Simulation engine
│   ├── risk_manager.py  # Position sizing
│   ├── alerts.py        # Notifications
│   └── utils.py         # Helpers
└── data/
    └── historical/      # Cached data
```

## Asset-Specific Parameters

Each asset has tuned parameters (found via `--auto-tune`):

```python
ASSET_CONFIGS = {
    'LINK/USD': {'range_threshold': 0.10, 'volume_threshold': 1.2},
    'SOL/USD': {'range_threshold': 0.0686, 'volume_threshold': 1.0},
    'XRP/USD': {'range_threshold': 0.0761, 'volume_threshold': 1.0},
    'BTC/USD': {'range_threshold': 0.04, 'volume_threshold': 1.2},
}
```

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk. Past performance does not guarantee future results. Always paper trade first and never risk more than you can afford to lose.
