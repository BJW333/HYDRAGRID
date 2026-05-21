"""
Consolidation Breakout Strategy - Configuration Settings
=========================================================
All tunable parameters in one place. Modify here, not in code.

FIXED VERSION - Changes marked with # FIXED
"""

# =============================================================================
# STRATEGY PARAMETERS
# =============================================================================

# Plateau Detection
LOOKBACK_PERIOD = 20              # Candles to check for squeeze
RANGE_THRESHOLD = 0.03            # FIXED: 3% (was 4%) - tighter squeeze = better breakouts
MIN_CONSOLIDATION_CANDLES = 20    # Minimum candles in consolidation

# Bollinger Bands
BB_LENGTH = 20                    # Period for BB calculation
BB_STD = 2.0                      # Standard deviations

# Trend Filter
TREND_MA_LENGTH = 50              # Moving average for trend direction
TREND_MA_SLOPE_LOOKBACK = 5       # Candles to measure MA slope

# Volume Confirmation
VOLUME_MA_LENGTH = 20             # Period for volume moving average
VOLUME_THRESHOLD = 1.8            # FIXED: 180% (was 120%) - real breakouts have big volume

# ATR (Average True Range)
ATR_PERIOD = 14                   # Period for ATR calculation

# =============================================================================
# RISK MANAGEMENT - THIS IS WHERE THE MONEY IS MADE/LOST
# =============================================================================

# FIXED: R:R ratio changed from 1.1 to 2.67
# Old: 2.2/2.0 = 1.1 R:R → needed 48% WR → LOSING
# New: 4.0/1.5 = 2.67 R:R → need 27% WR → PROFITABLE with 35-40% WR

STOP_LOSS_ATR = 1.5               # FIXED: tighter stop (was 2.0)
PROFIT_TARGET_ATR = 4.0           # FIXED: let winners run (was 2.2)
BREAKEVEN_TRIGGER_ATR = 2.0       # FIXED: don't move to BE too early (was 1.5)
TRAILING_STOP_ATR = 1.5           # FIXED: wider trail (was 1.0)

RISK_PER_TRADE = 0.10             # 10% per trade — aggressive for $1k account
MAX_DRAWDOWN = 0.40               # 40% kill switch — need room at higher risk

# =============================================================================
# DATA SETTINGS
# =============================================================================

# Primary trading pair
SYMBOL = "LINK/USD"               # Best performer from backtests
TIMEFRAME = "4h"                  # 1m, 5m, 15m, 30m, 1h, 4h, 1d

# Exchange backend: "kraken" (ccxt) or "alpaca"
EXCHANGE = "alpaca"               # Switch to "kraken" for ccxt/Kraken

# Assets to monitor with tuned parameters (used by live_trader.py)
# FIXED: Tighter range thresholds, higher volume requirements
ASSET_CONFIGS = {
    'SOL/USD': {'range_threshold': 0.06, 'volume_threshold': 1.8, 'enabled': True},
    'ETH/USD': {'range_threshold': 0.05, 'volume_threshold': 2.0, 'enabled': True},
    'LINK/USD': {'range_threshold': 0.06, 'volume_threshold': 1.8, 'enabled': True},
    'AVAX/USD': {'range_threshold': 0.06, 'volume_threshold': 1.8, 'enabled': True},
}

# Timeframe options
TIMEFRAMES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

# =============================================================================
# ACCOUNT SETTINGS
# =============================================================================

STARTING_CAPITAL = 1100           # Internal capital tracking (independent of Alpaca balance)
PAPER_TRADING = True              # True = paper, False = live

# =============================================================================
# EXCHANGE API (load from environment variables)
# =============================================================================

import os

# --- Alpaca (paper + live) ---
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET', '')

# --- Kraken (legacy ccxt) ---
KRAKEN_API_KEY = os.environ.get('KRAKEN_API_KEY', '')
KRAKEN_API_SECRET = os.environ.get('KRAKEN_API_SECRET', '')

# Active credentials (auto-selected by EXCHANGE)
if EXCHANGE == 'alpaca':
    API_KEY = ALPACA_API_KEY
    API_SECRET = ALPACA_API_SECRET
else:
    API_KEY = KRAKEN_API_KEY
    API_SECRET = KRAKEN_API_SECRET

# To set these, run:
#   export ALPACA_API_KEY="your-key-here"
#   export ALPACA_API_SECRET="your-secret-here"

# =============================================================================
# NOTIFICATIONS
# =============================================================================

ENABLE_NOTIFICATIONS = True
VERBOSE = True                    # Print to console

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# Discord (optional)
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')

# =============================================================================
# LOGGING
# =============================================================================

LOG_TRADES = True
LOG_FILE = "logs/trades.csv"

# =============================================================================
# BACKTESTING
# =============================================================================

BACKTEST_START = "2024-01-01"
BACKTEST_END = "2025-01-01"
COMMISSION = 0.0015 if EXCHANGE == 'alpaca' else 0.0026  # Alpaca 0.15% / Kraken 0.26%
SLIPPAGE = 0.0005                 # 0.05% estimated slippage
