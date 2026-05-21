#!/usr/bin/env python3
"""
Live Trader - Multi-Strategy Trading Bot
=========================================
Real-time trading bot that monitors multiple assets with multiple strategies.

Strategies:
    BREAKOUT:  4h consolidation breakout with BB squeeze detection
    MEAN_REV:  4h oversold bounce (RSI + Bollinger Band)
    HYDRA:     15m regime-gated multi-engine strategy

Usage:
    python live_trader.py                              # Paper trading, all strategies
    python live_trader.py --live                       # Live trading
    python live_trader.py --strategies breakout hydra  # Only specific strategies
    python live_trader.py --strategies hydra           # HYDRA only
    python live_trader.py --assets LINK/USD SOL/USD    # Specific assets
    python live_trader.py --interval 60                # Check every 60 seconds

Features:
    - Multi-asset monitoring with per-asset tuned parameters
    - Multi-strategy: BREAKOUT + MEAN_REV (4h) + HYDRA (15m)
    - Paper trading mode for testing
    - Live trading via Kraken API
    - Telegram/Discord alerts
    - Position tracking and P&L reporting
    - State persistence across restarts
    - HYDRA's TP1 → breakeven → TP2 exit management
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import traceback

sys.path.insert(0, str(Path(__file__).parent))

try:
    import ccxt
except ImportError:
    ccxt = None  # Not needed when using Alpaca
import pandas as pd
import numpy as np

from config.settings import (
    EXCHANGE, API_KEY, API_SECRET, STARTING_CAPITAL,
    RISK_PER_TRADE, STOP_LOSS_ATR, PROFIT_TARGET_ATR,
    TIMEFRAME, VERBOSE
)
from src.data_fetcher import DataFetcher
from src.indicators import add_all_indicators
from src.alerts import AlertManager
from src.risk_manager import RiskManager
from src.hydra_spot import (
    HydraConfig, check_hydra_signal
)
from src.grid_strategy import GridStrategy, GridConfig
from src.vwap_strategy import VWAPStrategy, VWAPConfig
from src.momentum_strategy import MomentumTrailer, MomentumConfig

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import asset-specific tuned parameters
from config.settings import ASSET_CONFIGS
ASSET_PARAMS = ASSET_CONFIGS

# Strategy timeframes
BREAKOUT_TIMEFRAME = '4h'
HYDRA_TIMEFRAME = '15m'

# Mean reversion parameters (per asset)
MEAN_REV_PARAMS = {
    'LINK/USD': {'rsi_oversold': 30, 'rsi_overbought': 70, 'enabled': True},
    'SOL/USD': {'rsi_oversold': 30, 'rsi_overbought': 70, 'enabled': True},
    'XRP/USD': {'rsi_oversold': 30, 'rsi_overbought': 70, 'enabled': True},
    'BTC/USD': {'rsi_oversold': 25, 'rsi_overbought': 75, 'enabled': True},
    'ETH/USD': {'rsi_oversold': 25, 'rsi_overbought': 75, 'enabled': True},
}

# HYDRA-specific config
HYDRA_PARAMS = {
    'LINK/USD': {'enabled': False},  # 6% WR, -$8.07 in live — worst performer
    'SOL/USD': {'enabled': True},
    'ETH/USD': {'enabled': True},
    'BTC/USD': {'enabled': True},
    'XRP/USD': {'enabled': True},
}

# Grid trading config — consistent income from oscillation
GRID_CONFIGS = {
    'SOL/USD': GridConfig(symbol='SOL/USD', num_grids=10, auto_range_pct=0.05, adaptive=True),
    'ETH/USD': GridConfig(symbol='ETH/USD', num_grids=10, auto_range_pct=0.05, adaptive=True),
    'LINK/USD': GridConfig(symbol='LINK/USD', num_grids=10, auto_range_pct=0.05, adaptive=True),
    'AVAX/USD': GridConfig(symbol='AVAX/USD', num_grids=10, auto_range_pct=0.05, adaptive=True),
}

# Volatility-weighted allocation (SOL oscillates most, BTC least)
VOL_WEIGHTS = {
    'SOL/USD': 0.30,
    'ETH/USD': 0.25,
    'LINK/USD': 0.25,
    'AVAX/USD': 0.20,
}

# =============================================================================
# POSITION DATACLASS
# =============================================================================

@dataclass
class Position:
    """Tracks an open position."""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    strategy: str   # 'BREAKOUT', 'MEAN_REV', or 'HYDRA'
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float
    take_profit: float  # Final TP (TP2 for HYDRA)
    atr_at_entry: float
    
    # Mean reversion specific
    bb_middle: float = 0.0
    
    # HYDRA specific
    tp1: float = 0.0           # First profit target
    tp2: float = 0.0           # Second profit target
    tp1_hit: bool = False      # Whether TP1 has been reached
    engine: str = ''           # HYDRA engine that generated signal
    quality_score: float = 0.0 # HYDRA quality score
    regime: str = ''           # HYDRA regime at entry
    entry_bar: int = 0         # For TTL tracking
    ttl_bars: int = 0          # Time-to-live in bars
    original_stop: float = 0.0 # Original stop before breakeven move
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.direction == 'LONG':
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'strategy': self.strategy,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'atr_at_entry': self.atr_at_entry,
            'bb_middle': self.bb_middle,
            'tp1': self.tp1,
            'tp2': self.tp2,
            'tp1_hit': self.tp1_hit,
            'engine': self.engine,
            'quality_score': self.quality_score,
            'regime': self.regime,
            'entry_bar': self.entry_bar,
            'ttl_bars': self.ttl_bars,
            'original_stop': self.original_stop,
        }


# =============================================================================
# LIVE TRADER
# =============================================================================

class LiveTrader:
    """
    Multi-strategy live trading bot.
    
    Strategies:
        BREAKOUT: 4h consolidation breakout with BB squeeze
        MEAN_REV: 4h oversold bounce (RSI + BB)
        HYDRA: 15m regime-gated multi-engine strategy
    """
    
    def __init__(self, 
                 assets: List[str] = None,
                 strategies: List[str] = None,
                 exchange_id: str = EXCHANGE,
                 capital: float = STARTING_CAPITAL,
                 risk_per_trade: float = RISK_PER_TRADE,
                 paper_mode: bool = True,
                 api_key: str = API_KEY,
                 api_secret: str = API_SECRET):
        
        # Assets to trade
        self.assets = assets or [a for a, p in ASSET_PARAMS.items() if p.get('enabled', True)]
        
        # Strategies to run
        self.enabled_strategies = strategies or ['grid']
        self.enabled_strategies = [s.lower() for s in self.enabled_strategies]
        
        self.exchange_id = exchange_id
        self.capital = capital
        self.starting_capital = capital
        self.risk_per_trade = risk_per_trade
        self.paper_mode = paper_mode
        
        # Initialize components
        self.fetcher = DataFetcher(exchange_id=exchange_id)
        self.alerts = AlertManager(enable_console=True)
        self.risk_manager = RiskManager(
            initial_capital=capital,
            risk_per_trade=risk_per_trade
        )
        
        # HYDRA strategy config
        self.hydra_config = HydraConfig()
        
        # Connect to exchange
        # For Alpaca, always connect — paper trading uses the paper API endpoint
        if not paper_mode or exchange_id == 'alpaca':
            self.exchange = self._connect_exchange(api_key, api_secret)
        else:
            self.exchange = None
        
        # State tracking
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_trades: List[dict] = []
        self.running = False
        
        # Data cache (separate for each timeframe)
        self.data_cache_4h: Dict[str, pd.DataFrame] = {}
        self.data_cache_15m: Dict[str, pd.DataFrame] = {}
        
        # HYDRA bar tracking for TTL
        self.hydra_bar_index: Dict[str, int] = {}  # symbol -> current bar index
        self._last_15m_timestamp: Dict[str, pd.Timestamp] = {}  # Track last seen timestamp
        
        # Signal deduplication: prevents re-entering the same signal on the same candle
        # Key: (symbol, strategy) -> candle timestamp that last generated a signal
        self._last_signal_candle: Dict[tuple, object] = {}
        
        # Daily trade limit safety (prevents runaway loops)
        self._daily_trade_count: int = 0
        self._daily_trade_date: str = ''
        self.MAX_DAILY_TRADES = 20  # kill switch if something goes wrong
        
        # Grid trading strategies
        self.grids: Dict[str, GridStrategy] = {}
        if 'grid' in self.enabled_strategies and self.exchange:
            grid_capital = capital * 0.85  # 85% of capital to grids
            for symbol, gcfg in GRID_CONFIGS.items():
                weight = VOL_WEIGHTS.get(symbol, 1.0 / len(GRID_CONFIGS))
                gcfg.total_investment = grid_capital * weight
                print(f"Grid {symbol}: ${gcfg.total_investment:.2f} ({weight*100:.0f}% allocation)")
                grid = GridStrategy(self.exchange, gcfg, self.alerts)
                # Try loading saved state first
                if not grid.load_state():
                    # Fresh start — will setup when we get current price
                    pass
                self.grids[symbol] = grid
        
        # VWAP reversion strategies
        self.vwap_strategies: Dict[str, VWAPStrategy] = {}
        self.data_cache_5m: Dict[str, pd.DataFrame] = {}
        if 'vwap' in self.enabled_strategies:
            for symbol in self.assets:
                vwap = VWAPStrategy(capital=capital, alerts=self.alerts)
                vwap.load_state()
                self.vwap_strategies[symbol] = vwap
        
        # Momentum breakout trailer
        self.momentum_strategies: Dict[str, MomentumTrailer] = {}
        if 'momentum' in self.enabled_strategies:
            for symbol in self.assets:
                mt = MomentumTrailer(capital=capital, alerts=self.alerts)
                mt.load_state()
                self.momentum_strategies[symbol] = mt
                            
        # State file for persistence
        self.state_file = Path(__file__).parent / "data" / "trader_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_state()
    
    def _connect_exchange(self, api_key: str, api_secret: str):
        """Connect to exchange with API credentials."""
        if not api_key or not api_secret:
            raise ValueError("API key and secret required for live trading")
        
        if self.exchange_id == 'alpaca':
            from src.alpaca_adapter import AlpacaExchange
            exchange = AlpacaExchange({
                'apiKey': api_key,
                'secret': api_secret,
                'paper': self.paper_mode,
            })
        else:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
        
        # Test connection
        try:
            balance = exchange.fetch_balance()
            usd_balance = balance.get('USD', {}).get('free', 0)
            self.alerts.status_alert(f"Connected to {self.exchange_id}. USD Balance: ${usd_balance:,.2f}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to exchange: {e}")
        
        return exchange
    
    def _load_state(self):
        """Load persisted state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore positions
                for symbol, pos_data in state.get('positions', {}).items():
                    self.positions[symbol] = Position(
                        symbol=pos_data['symbol'],
                        direction=pos_data['direction'],
                        strategy=pos_data.get('strategy', 'BREAKOUT'),
                        entry_price=pos_data['entry_price'],
                        entry_time=datetime.fromisoformat(pos_data['entry_time']),
                        size=pos_data['size'],
                        stop_loss=pos_data['stop_loss'],
                        take_profit=pos_data['take_profit'],
                        atr_at_entry=pos_data.get('atr_at_entry', 0),
                        bb_middle=pos_data.get('bb_middle', 0.0),
                        tp1=pos_data.get('tp1', 0.0),
                        tp2=pos_data.get('tp2', 0.0),
                        tp1_hit=pos_data.get('tp1_hit', False),
                        engine=pos_data.get('engine', ''),
                        quality_score=pos_data.get('quality_score', 0.0),
                        regime=pos_data.get('regime', ''),
                        entry_bar=pos_data.get('entry_bar', 0),
                        ttl_bars=pos_data.get('ttl_bars', 0),
                        original_stop=pos_data.get('original_stop', 0.0),
                    )
                
                self.closed_trades = state.get('closed_trades', [])
                self.capital = state.get('capital', self.starting_capital)
                self.hydra_bar_index = state.get('hydra_bar_index', {})
                
                if self.positions:
                    self.alerts.status_alert(f"Restored {len(self.positions)} open positions")
                    
            except Exception as e:
                self.alerts.error_alert(f"Failed to load state: {e}")
                if not self.paper_mode:
                    raise RuntimeError(f"Cannot start live trading without state: {e}")
                
    def _save_state(self):
        """Persist state to file."""
        state = {
            'positions': {s: p.to_dict() for s, p in self.positions.items()},
            'closed_trades': self.closed_trades[-100:],  # Keep last 100
            'capital': self.capital,
            'hydra_bar_index': self.hydra_bar_index,
            'last_update': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_data_4h(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """Fetch 4h candles for BREAKOUT/MEAN_REV strategies."""
        try:
            df = self.fetcher.fetch_live(
                symbol=symbol,
                timeframe=BREAKOUT_TIMEFRAME,
                num_candles=lookback
            )
            
            if len(df) > 0:
                df = add_all_indicators(df)
                df = self._add_mean_reversion_indicators(df)
                self.data_cache_4h[symbol] = df
            
            return df
            
        except Exception as e:
            self.alerts.error_alert(f"Error fetching 4h {symbol}: {e}")
            return self.data_cache_4h.get(symbol, pd.DataFrame())
    
    def fetch_data_15m(self, symbol: str, lookback: int = 200) -> pd.DataFrame:
        """Fetch 15m candles for HYDRA strategy."""
        try:
            df = self.fetcher.fetch_live(
                symbol=symbol,
                timeframe=HYDRA_TIMEFRAME,
                num_candles=lookback
            )
            
            if len(df) > 0:
                last_timestamp = df.index[-1] if hasattr(df.index[-1], 'timestamp') else df.index[-1]
                prev_timestamp = self._last_15m_timestamp.get(symbol)
                
                if prev_timestamp is None or last_timestamp > prev_timestamp:
                    self.hydra_bar_index[symbol] = self.hydra_bar_index.get(symbol, 0) + 1
                    self._last_15m_timestamp[symbol] = last_timestamp
                
                self.data_cache_15m[symbol] = df
            
            return df
            
        except Exception as e:
            self.alerts.error_alert(f"Error fetching 15m {symbol}: {e}")
            return self.data_cache_15m.get(symbol, pd.DataFrame())
    
    def _add_mean_reversion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI and mean reversion signals."""
        # RSI calculation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # BB position (0 = at lower band, 1 = at upper band)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Momentum (5-period return)
        df['momentum'] = df['close'].pct_change(periods=5) * 100
        
        return df
    
    # =========================================================================
    # BREAKOUT STRATEGY (4h)
    # =========================================================================
    
    def check_breakout_signal(self, symbol: str, df: pd.DataFrame) -> Optional[str]:
        """
        Check for BREAKOUT entry signal.
        
        Returns:
            'LONG' or None
        """
        if len(df) < 50:
            return None
        
        # Get asset-specific parameters
        params = ASSET_PARAMS.get(symbol, {
            'range_threshold': 0.05,
            'volume_threshold': 1.2
        })
        
        range_threshold = params['range_threshold']
        volume_threshold = params['volume_threshold']
        
        # Get last CLOSED candle (iloc[-1] may be incomplete)
        current = df.iloc[-2]
        previous = df.iloc[-3]
        
        # Check plateau condition (was active on previous candle)
        bb_squeeze_prev = previous['bb_width_percentile'] < 25
        tight_range_prev = previous['price_range'] < range_threshold
        plateau_was_active = bb_squeeze_prev and tight_range_prev
        
        if not plateau_was_active:
            return None
        
        # Check trend alignment
        long_trend = current['ma_slope'] > 0 and current['close'] > current['ma_50']
        
        # Check breakout
        prev_range_high = previous['range_high']
        volume_ok = current['volume_ratio'] >= volume_threshold
        long_breakout = current['high'] > prev_range_high and volume_ok
        
        # Generate signal
        if plateau_was_active and long_trend and long_breakout:
            return 'LONG'
        
        return None
    
    # =========================================================================
    # MEAN REVERSION STRATEGY (4h)
    # =========================================================================
    
    def check_mean_reversion_signal(self, symbol: str, df: pd.DataFrame) -> Optional[str]:
        """
        Check for MEAN REVERSION entry signal (buy the dip).
        
        Returns:
            'LONG' or None
        """
        if len(df) < 25:
            return None
        
        # Get asset-specific parameters
        params = MEAN_REV_PARAMS.get(symbol, {'rsi_oversold': 30, 'enabled': True})
        if not params.get('enabled', True):
            return None
        
        rsi_oversold = params['rsi_oversold']
        
        current = df.iloc[-2]
        previous = df.iloc[-3]
        
        # Check oversold conditions
        rsi_oversold_now = current['rsi'] < rsi_oversold
        at_lower_bb = current['bb_position'] < 0.2
        not_freefall = current['momentum'] > -10.0  # Not crashing too hard
        rsi_recovering = current['rsi'] > previous['rsi']  # Starting to bounce
        
        if rsi_oversold_now and at_lower_bb and not_freefall and rsi_recovering:
            return 'LONG'
        
        return None
    
    # =========================================================================
    # HYDRA STRATEGY (15m)
    # =========================================================================
    
    def check_hydra_signal(self, symbol: str, df: pd.DataFrame) -> Optional[dict]:
        """
        Check for HYDRA entry signal.
        
        Returns:
            Signal dict with direction, engine, entry, stop, tp1, tp2, quality, regime, atr
            or None
        """
        if len(df) < 150:
            return None
        
        # Check if HYDRA is enabled for this asset
        params = HYDRA_PARAMS.get(symbol, {'enabled': True})
        if not params.get('enabled', True):
            return None
        
        # Use the convenience function from hydra_spot
        signal = check_hydra_signal(df, symbol, self.hydra_config)
        
        return signal
    
    # =========================================================================
    # EXIT MANAGEMENT
    # =========================================================================
    
    def check_exit_conditions_4h(self, symbol: str, df: pd.DataFrame) -> Optional[str]:
        """
        Check if 4h position (BREAKOUT/MEAN_REV) should be closed.
        
        Returns:
            'STOP_LOSS', 'TAKE_PROFIT', 'RSI_NORMALIZED', 'MEAN_REACHED', or None
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Only check 4h positions
        if position.strategy == 'HYDRA':
            return None
        
        current = df.iloc[-2]  # Last closed candle
        current_price = current['close']
        high = current['high']
        low = current['low']
        
        if position.direction == 'LONG':
            # Check stop loss
            if low <= position.stop_loss:
                return 'STOP_LOSS'
            # Check take profit
            if high >= position.take_profit:
                return 'TAKE_PROFIT'
            
            # Mean reversion specific exits
            if position.strategy == 'MEAN_REV':
                # Exit when RSI normalizes above 50
                if current['rsi'] > 50:
                    return 'RSI_NORMALIZED'
                # Exit when price reaches middle BB (the mean)
                if current_price >= current['bb_middle']:
                    return 'MEAN_REACHED'
        
        else:  # SHORT
            if high >= position.stop_loss:
                return 'STOP_LOSS'
            if low <= position.take_profit:
                return 'TAKE_PROFIT'
        
        return None
    
    def check_exit_conditions_hydra(self, symbol: str, df: pd.DataFrame) -> Optional[str]:
        """
        Check HYDRA-specific exit conditions with TP1 → breakeven → TP2 logic.
        
        Returns:
            'STOP', 'TP1', 'TP2', 'TTL', or None
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Only check HYDRA positions
        if position.strategy != 'HYDRA':
            return None
        
        current = df.iloc[-2]
        high = current['high']
        low = current['low']
        close = current['close']
        
        if position.direction == 'LONG':
            # Check stop loss
            if low <= position.stop_loss:
                return 'STOP'
            
            # Check TP1 (move stop to breakeven)
            if not position.tp1_hit and high >= position.tp1:
                # Mark TP1 as hit and move stop to breakeven
                position.tp1_hit = True
                position.original_stop = position.stop_loss
                position.stop_loss = position.entry_price  # Move to breakeven
                self._save_state()
                
                self.alerts.send(
                    f"🎯 HYDRA TP1 HIT: {position.symbol}\n"
                    f"Stop moved to breakeven @ ${position.entry_price:,.2f}",
                    level="INFO"
                )
                # Don't exit yet - let it run to TP2
                return None
            
            # Check TP2 (final exit)
            if high >= position.tp2:
                return 'TP2'
            
            # Check TTL expiry
            current_bar = self.hydra_bar_index.get(symbol, 0)
            bars_held = current_bar - position.entry_bar
            if position.ttl_bars > 0 and bars_held >= position.ttl_bars:
                # Only exit if significantly underwater
                pnl_pct = (close - position.entry_price) / position.entry_price
                if pnl_pct < -0.005:  # More than 0.5% underwater
                    return 'TTL'
        
        else:  # SHORT
            if high >= position.stop_loss:
                return 'STOP'
            
            if not position.tp1_hit and low <= position.tp1:
                position.tp1_hit = True
                position.original_stop = position.stop_loss
                position.stop_loss = position.entry_price
                self._save_state()
                
                self.alerts.send(
                    f"🎯 HYDRA TP1 HIT: {position.symbol}\n"
                    f"Stop moved to breakeven @ ${position.entry_price:,.2f}",
                    level="INFO"
                )
                return None
            
            if low <= position.tp2:
                return 'TP2'
            
            current_bar = self.hydra_bar_index.get(symbol, 0)
            bars_held = current_bar - position.entry_bar
            if position.ttl_bars > 0 and bars_held >= position.ttl_bars:
                pnl_pct = (position.entry_price - close) / position.entry_price
                if pnl_pct < -0.005:
                    return 'TTL'
        
        return None
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk, capped to available capital."""
        risk_amount = self.capital * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        size = risk_amount / risk_per_unit
        
        # Check total exposure including existing positions
        current_exposure = sum(p.size * p.entry_price for p in self.positions.values())
        max_new_exposure = (self.capital * 0.95) - current_exposure
        max_size = max_new_exposure / entry_price
        
        # Cap size to both risk-based and exposure-based limits
        size = min(size, max_size)
        
        return max(0, size)  # Ensure non-negative
    
    def open_position(self, symbol: str, direction: str, strategy: str, 
                      df: pd.DataFrame, signal: dict = None):
        """
        Open a new position.
        
        Args:
            symbol: Trading pair
            direction: 'LONG' or 'SHORT'
            strategy: 'BREAKOUT', 'MEAN_REV', or 'HYDRA'
            df: DataFrame with indicators
            signal: HYDRA signal dict (optional, only for HYDRA)
        """
        current = df.iloc[-2]  # Last closed candle
        entry_price = current['close']
        
        # Strategy-specific stop/target calculation
        if strategy == 'HYDRA' and signal:
            # Use HYDRA's calculated levels
            entry_price = signal['entry']
            stop_loss = signal['stop']
            tp1 = signal['tp1']
            tp2 = signal['tp2']
            take_profit = tp2  # Final TP
            atr = signal.get('atr', 0)
            
            position = Position(
                symbol=symbol,
                direction=direction,
                strategy=strategy,
                entry_price=entry_price,
                entry_time=datetime.now(),
                size=0,  # Calculate below
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_at_entry=atr,
                tp1=tp1,
                tp2=tp2,
                tp1_hit=False,
                engine=signal.get('engine', ''),
                quality_score=signal.get('quality', 0),
                regime=str(signal.get('regime', '')),
                entry_bar=self.hydra_bar_index.get(symbol, 0),
                ttl_bars=self.hydra_config.compress_ttl,  # Use default TTL
                original_stop=stop_loss,
            )
            
        elif strategy == 'BREAKOUT':
            atr = current['atr']
            stop_loss = entry_price - (atr * STOP_LOSS_ATR)
            take_profit = entry_price + (atr * PROFIT_TARGET_ATR)
            
            position = Position(
                symbol=symbol,
                direction=direction,
                strategy=strategy,
                entry_price=entry_price,
                entry_time=datetime.now(),
                size=0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_at_entry=atr,
                bb_middle=current['bb_middle'],
            )
            
        elif strategy == 'MEAN_REV':
            atr = current['atr']
            stop_loss = entry_price - (atr * 1.5)  # Tighter stop
            take_profit = entry_price + (atr * 2.0)  # Or exit at mean
            
            position = Position(
                symbol=symbol,
                direction=direction,
                strategy=strategy,
                entry_price=entry_price,
                entry_time=datetime.now(),
                size=0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_at_entry=atr,
                bb_middle=current['bb_middle'],
            )
        else:
            # Fallback
            atr = current.get('atr', entry_price * 0.02)
            stop_loss = entry_price - (atr * STOP_LOSS_ATR)
            take_profit = entry_price + (atr * PROFIT_TARGET_ATR)
            
            position = Position(
                symbol=symbol,
                direction=direction,
                strategy=strategy,
                entry_price=entry_price,
                entry_time=datetime.now(),
                size=0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_at_entry=atr,
            )
        
        # Calculate size
        size = self.calculate_position_size(entry_price, position.stop_loss)
        
        if size <= 0:
            self.alerts.error_alert(f"Invalid position size for {symbol}")
            return
        
        position.size = size
        
        # Execute trade
        if not self.paper_mode:
            try:
                order = self._execute_order(symbol, direction, size)
                position.entry_price = order.get('average', entry_price)
            except Exception as e:
                self.alerts.error_alert(f"Order failed: {e}")
                return
        
        self.positions[symbol] = position
        self._save_state()
        
        # Alert with strategy info
        if strategy == 'HYDRA':
            strategy_emoji = "🐉"
            extra_info = f"Engine: {position.engine} | Quality: {position.quality_score:.2f}"
        elif strategy == 'BREAKOUT':
            strategy_emoji = "💥"
            extra_info = ""
        else:
            strategy_emoji = "🔄"
            extra_info = ""
        
        self.alerts.send(
            f"{strategy_emoji} {strategy} SIGNAL: {direction} {symbol}\n"
            f"Entry: ${position.entry_price:,.2f} | Stop: ${position.stop_loss:,.2f} | "
            f"Target: ${position.take_profit:,.2f}\n{extra_info}",
            level="SIGNAL"
        )
        self.alerts.trade_alert(symbol, "OPEN", direction, position.entry_price, size)
    
    def close_position(self, symbol: str, reason: str, df: pd.DataFrame):
        """Close an open position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current = df.iloc[-2]  # Use last CLOSED candle for consistent pricing
        
        # Determine exit price based on reason
        if reason in ['STOP_LOSS', 'STOP']:
            exit_price = position.stop_loss
        elif reason == 'TAKE_PROFIT':
            exit_price = position.take_profit
        elif reason == 'TP2':
            exit_price = position.tp2
        elif reason == 'TP1':
            exit_price = position.tp1
        else:
            exit_price = current['close']
        
        # Execute trade
        if not self.paper_mode:
            try:
                close_direction = 'SHORT' if position.direction == 'LONG' else 'LONG'
                order = self._execute_order(symbol, close_direction, position.size)
                exit_price = order.get('average', exit_price)
            except Exception as e:
                self.alerts.error_alert(f"Close order failed: {e}")
                return
        
        # Calculate P&L
        pnl = position.unrealized_pnl(exit_price)
        
        # Simulate fees in paper mode too
        if self.paper_mode:
            from config.settings import COMMISSION
            commission = (position.entry_price * position.size * COMMISSION) + \
                        (exit_price * position.size * COMMISSION)
            pnl -= commission
        
        self.capital += pnl
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'strategy': position.strategy,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time.isoformat(),
            'exit_time': datetime.now().isoformat(),
            'size': position.size,
            'pnl': pnl,
            'exit_reason': reason,
            'engine': position.engine,  # HYDRA
            'quality_score': position.quality_score,  # HYDRA
            'tp1_hit': position.tp1_hit,  # HYDRA
        }
        self.closed_trades.append(trade_record)
        
        # Remove position
        del self.positions[symbol]
        self._save_state()
        
        # Alert
        if position.strategy == 'HYDRA':
            strategy_emoji = "🐉"
        elif position.strategy == 'BREAKOUT':
            strategy_emoji = "💥"
        else:
            strategy_emoji = "🔄"
        
        pnl_emoji = "✅" if pnl > 0 else "❌"
        self.alerts.send(
            f"{pnl_emoji} CLOSED [{position.strategy}] {symbol}\n"
            f"P&L: ${pnl:,.2f} | Reason: {reason}",
            level="TRADE"
        )
    
    def _execute_order(self, symbol: str, direction: str, size: float) -> dict:
        """Execute a market order on the exchange."""
        side = 'buy' if direction == 'LONG' else 'sell'
        
        order = self.exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=size
        )
        
        return order
    
    # =========================================================================
    # SIGNAL DEDUP & SAFETY
    # =========================================================================
    
    def _is_duplicate_signal(self, symbol: str, strategy: str, df: pd.DataFrame) -> bool:
        """
        Check if we already traded this signal candle.
        Prevents the death-loop bug where a 4h signal fires every 60s.
        """
        candle_ts = df.iloc[-2].name  # Last CLOSED candle timestamp
        key = (symbol, strategy)
        
        if key in self._last_signal_candle and self._last_signal_candle[key] == candle_ts:
            return True  # Already traded this candle
        
        # Record this signal
        self._last_signal_candle[key] = candle_ts
        return False
    
    def _check_daily_limit(self) -> bool:
        """
        Returns True if daily trade limit has been hit.
        Resets at midnight.
        """
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self._daily_trade_date:
            self._daily_trade_date = today
            self._daily_trade_count = 0
        
        if self._daily_trade_count >= self.MAX_DAILY_TRADES:
            self.alerts.error_alert(
                f"⛔ Daily trade limit ({self.MAX_DAILY_TRADES}) reached. "
                f"Skipping new entries until tomorrow."
            )
            return True
        
        self._daily_trade_count += 1
        return False
    
    # =========================================================================
    # MAIN TRADING LOOP
    # =========================================================================
    
    def run_4h_strategies(self):
        """Run BREAKOUT and MEAN_REV strategies (4h timeframe)."""
        if 'breakout' not in self.enabled_strategies and 'mean_rev' not in self.enabled_strategies:
            return
        
        for symbol in self.assets:
            try:
                # Fetch 4h data
                df = self.fetch_data_4h(symbol)
                
                if len(df) == 0:
                    continue
                
                # Check for exits first (4h positions only)
                if symbol in self.positions and self.positions[symbol].strategy in ['BREAKOUT', 'MEAN_REV']:
                    exit_reason = self.check_exit_conditions_4h(symbol, df)
                    if exit_reason:
                        self.close_position(symbol, exit_reason, df)
                        continue
                
                # Skip if already in position for this symbol
                if symbol in self.positions:
                    continue
                
                # Check for BREAKOUT signal (higher priority)
                if 'breakout' in self.enabled_strategies:
                    breakout_signal = self.check_breakout_signal(symbol, df)
                    if breakout_signal:
                        if self._is_duplicate_signal(symbol, 'BREAKOUT', df):
                            continue
                        if self._check_daily_limit():
                            continue
                        self.open_position(symbol, breakout_signal, 'BREAKOUT', df)
                        continue
                
                # Check for MEAN REVERSION signal
                if 'mean_rev' in self.enabled_strategies:
                    mr_signal = self.check_mean_reversion_signal(symbol, df)
                    if mr_signal:
                        if self._is_duplicate_signal(symbol, 'MEAN_REV', df):
                            continue
                        if self._check_daily_limit():
                            continue
                        self.open_position(symbol, mr_signal, 'MEAN_REV', df)
                        
            except Exception as e:
                self.alerts.error_alert(f"Error processing 4h {symbol}: {e}")
                traceback.print_exc()
    
    def run_hydra_strategy(self):
        """Run HYDRA strategy (15m timeframe)."""
        if 'hydra' not in self.enabled_strategies:
            return
        
        for symbol in self.assets:
            try:
                # Fetch 15m data
                df = self.fetch_data_15m(symbol)
                
                if len(df) < 150:
                    continue
                
                # Check for exits first (HYDRA positions only)
                if symbol in self.positions and self.positions[symbol].strategy == 'HYDRA':
                    exit_reason = self.check_exit_conditions_hydra(symbol, df)
                    if exit_reason:
                        self.close_position(symbol, exit_reason, df)
                        continue
                
                # Skip if already in position for this symbol
                if symbol in self.positions:
                    continue
                
                # Check for HYDRA signal
                signal = self.check_hydra_signal(symbol, df)
                if signal:
                    if self._is_duplicate_signal(symbol, 'HYDRA', df):
                        continue
                    if self._check_daily_limit():
                        continue
                    self.open_position(symbol, signal['direction'], 'HYDRA', df, signal)
                    
            except Exception as e:
                self.alerts.error_alert(f"Error processing HYDRA {symbol}: {e}")
                traceback.print_exc()
    
    def run_grid_strategy(self):
        """Run grid trading strategy — check fills, manage orders, adaptive rebalance."""
        if 'grid' not in self.enabled_strategies or not self.grids:
            return
        
        for symbol, grid in self.grids.items():
            try:
                ticker = self.fetcher.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                if current_price <= 0:
                    continue
                
                # Initialize grid if not yet active
                if not grid.active:
                    # Fetch candles for adaptive range calculation
                    if grid.config.adaptive:
                        try:
                            df_15m = self.fetcher.fetch_live(symbol, timeframe='15m', num_candles=100)
                            new_range = grid.calculate_adaptive_range(df_15m)
                            grid.config.auto_range_pct = new_range
                        except Exception:
                            pass  # Use default range if fetch fails
                    grid.setup(current_price)
                    continue
                
                # Check for filled orders
                fills = grid.check_fills()
                
                # Adaptive rebalance — check if price near grid edge
                if grid.config.adaptive and grid.needs_rebalance(current_price):
                    try:
                        df_15m = self.fetcher.fetch_live(symbol, timeframe='15m', num_candles=100)
                        grid.rebalance(current_price, df_15m)
                    except Exception as e:
                        print(f"Rebalance failed for {symbol}: {e}")
                        grid.rebalance(current_price)  # Rebalance without ATR data
                else:
                    # Normal safety check
                    grid.check_safety(current_price)
                    
            except Exception as e:
                self.alerts.error_alert(f"Error processing GRID {symbol}: {e}")
                traceback.print_exc()
                
    def fetch_data_5m(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """Fetch 5-minute candles for VWAP strategy."""
        try:
            df = self.fetcher.fetch_live(symbol=symbol, timeframe='5m', num_candles=lookback)
            if len(df) > 0:
                self.data_cache_5m[symbol] = df
            return df
        except Exception as e:
            self.alerts.error_alert(f"Error fetching 5m {symbol}: {e}")
            return self.data_cache_5m.get(symbol, pd.DataFrame())
    
    def run_vwap_strategy(self):
        """Run VWAP mean reversion strategy (5m timeframe)."""
        if 'vwap' not in self.enabled_strategies or not self.vwap_strategies:
            return
        
        for symbol, vwap in self.vwap_strategies.items():
            try:
                df = self.fetch_data_5m(symbol)
                if len(df) < 60:
                    continue
                
                current_price = df.iloc[-1]['close']
                
                # Check exit first
                if vwap.active_trade:
                    reason = vwap.check_exit(current_price)
                    if reason:
                        vwap.close_trade(current_price, reason)
                    continue
                
                # Check for new signal
                signal = vwap.check_signal(df, symbol)
                if signal:
                    vwap.open_trade(signal, symbol)
                    
            except Exception as e:
                self.alerts.error_alert(f"Error processing VWAP {symbol}: {e}")
                traceback.print_exc()
                
    def run_momentum_strategy(self):
        """Run momentum breakout trailer (5m timeframe)."""
        if 'momentum' not in self.enabled_strategies or not self.momentum_strategies:
            return
        
        for symbol, mt in self.momentum_strategies.items():
            try:
                df = self.fetch_data_5m(symbol)
                if len(df) < 60:
                    continue
                
                current_price = df.iloc[-1]['close']
                
                if mt.active_trade:
                    reason = mt.update_and_check_exit(current_price)
                    if reason:
                        mt.close_trade(current_price, reason)
                    continue
                
                signal = mt.check_signal(df, symbol)
                if signal:
                    mt.open_trade(signal, symbol)
                    
            except Exception as e:
                self.alerts.error_alert(f"Error processing MOMENTUM {symbol}: {e}")
                traceback.print_exc()
                
    def get_next_candle_time(self, timeframe: str) -> datetime:
        """Calculate when the next candle closes."""
        now = datetime.now()
        
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '1d': 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 240)
        
        current_minute = now.hour * 60 + now.minute
        next_close = ((current_minute // minutes) + 1) * minutes
        
        # Handle day overflow
        days_ahead = next_close // 1440
        next_close = next_close % 1440
        
        next_time = now.replace(
            hour=next_close // 60,
            minute=next_close % 60,
            second=5,
            microsecond=0
        )
        
        if days_ahead > 0 or next_time <= now:
            next_time += timedelta(days=max(1, days_ahead))
        
        if next_time <= now:
            next_time += timedelta(days=1)
        
        return next_time
    
    def print_status(self):
        """Print current status."""
        print("\n" + "=" * 70)
        print(f"📊 MULTI-STRATEGY TRADER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        mode = "🟡 PAPER" if self.paper_mode else "🟢 LIVE"
        print(f"\n   Mode: {mode}")
        print(f"   Capital: ${self.capital:,.2f}")
        print(f"   P&L: ${self.capital - self.starting_capital:,.2f} ({((self.capital/self.starting_capital)-1)*100:.2f}%)")
        print(f"   Assets: {', '.join(self.assets)}")
        print(f"   Strategies: {', '.join(s.upper() for s in self.enabled_strategies)}")
        
        print(f"\n📈 OPEN POSITIONS ({len(self.positions)}):")
        if self.positions:
            for symbol, pos in self.positions.items():
                # Get current price from cache
                if pos.strategy == 'HYDRA':
                    df = self.data_cache_15m.get(symbol)
                else:
                    df = self.data_cache_4h.get(symbol)
                
                if df is not None and len(df) > 0:
                    current_price = df.iloc[-1]['close']
                    pnl = pos.unrealized_pnl(current_price)
                    pnl_pct = (pnl / (pos.entry_price * pos.size)) * 100
                else:
                    pnl = 0
                    pnl_pct = 0
                
                dir_emoji = "🟢" if pos.direction == "LONG" else "🔴"
                if pos.strategy == "HYDRA":
                    strat_emoji = "🐉"
                elif pos.strategy == "BREAKOUT":
                    strat_emoji = "💥"
                else:
                    strat_emoji = "🔄"
                pnl_emoji = "✅" if pnl > 0 else "❌"
                
                print(f"   {dir_emoji} {strat_emoji} [{pos.strategy}] {symbol}: {pos.direction} @ ${pos.entry_price:,.2f}")
                print(f"      Size: {pos.size:.4f} | P&L: {pnl_emoji} ${pnl:,.2f} ({pnl_pct:.1f}%)")
                print(f"      SL: ${pos.stop_loss:,.2f} | TP: ${pos.take_profit:,.2f}")
                
                if pos.strategy == 'HYDRA':
                    tp1_status = "✓" if pos.tp1_hit else "○"
                    print(f"      Engine: {pos.engine} | TP1 [{tp1_status}]: ${pos.tp1:,.2f} | Quality: {pos.quality_score:.2f}")
        else:
            print("   No open positions")
        
        # Strategy breakdown of closed trades
        print(f"\n📜 RECENT TRADES ({len(self.closed_trades)} total):")
        for trade in self.closed_trades[-5:]:
            emoji = "✅" if trade['pnl'] > 0 else "❌"
            strat = trade.get('strategy', 'BREAKOUT')
            if strat == 'HYDRA':
                strat_emoji = "🐉"
            elif strat == 'BREAKOUT':
                strat_emoji = "💥"
            else:
                strat_emoji = "🔄"
            print(f"   {emoji} {strat_emoji} [{strat}] {trade['symbol']}: ${trade['pnl']:,.2f} ({trade['exit_reason']})")
        
        # P&L by strategy
        if self.closed_trades:
            print(f"\n📊 P&L BY STRATEGY:")
            for strat in ['BREAKOUT', 'MEAN_REV', 'HYDRA']:
                strat_trades = [t for t in self.closed_trades if t.get('strategy') == strat]
                if strat_trades:
                    total_pnl = sum(t['pnl'] for t in strat_trades)
                    winners = len([t for t in strat_trades if t['pnl'] > 0])
                    win_rate = (winners / len(strat_trades)) * 100 if strat_trades else 0
                    print(f"   {strat}: ${total_pnl:,.2f} ({len(strat_trades)} trades, {win_rate:.0f}% win)")
        
        # Grid status
        if self.grids:
            print(f"\n📐 GRID TRADING:")
            for symbol, grid in self.grids.items():
                s = grid.get_status()
                status_icon = "🟢" if s['active'] else "⚪"
                print(f"   {status_icon} {s['symbol']}: {s['range']} ({s['levels']} levels)")
                print(f"      Orders: {s['pending_buys']}B / {s['pending_sells']}S | "
                      f"Cycles: {s['completed']} | Profit: ${s['profit']:,.2f}")
        
        # VWAP status
        if self.vwap_strategies:
            print(f"\n🔮 VWAP REVERSION:")
            for symbol, vwap in self.vwap_strategies.items():
                s = vwap.get_status()
                if s['active_trade']:
                    print(f"   🟡 {symbol}: {s['direction']} active | "
                          f"Trades: {s['completed']} ({s['win_rate']:.0f}% WR) | "
                          f"Profit: ${s['profit']:,.2f}")
                else:
                    print(f"   ⚪ {symbol}: Watching | "
                          f"Trades: {s['completed']} ({s['win_rate']:.0f}% WR) | "
                          f"Profit: ${s['profit']:,.2f}")
        
        # Momentum status
        if self.momentum_strategies:
            print(f"\n🚀 MOMENTUM TRAILER:")
            for symbol, mt in self.momentum_strategies.items():
                s = mt.get_status()
                if s['active_trade']:
                    print(f"   🟡 {symbol}: LONG @ ${s['entry']:,.2f} | "
                          f"Peak: ${s['peak']:,.2f} | Trail: ${s['trail']:,.2f} | "
                          f"Profit: ${s['profit']:,.2f}")
                else:
                    print(f"   ⚪ {symbol}: Watching | "
                          f"Trades: {s['completed']} ({s['win_rate']:.0f}% WR) | "
                          f"Profit: ${s['profit']:,.2f}")
                    
        print("=" * 70)
    
    def run_once(self):
        """Run one iteration of all strategies."""
        # Check max drawdown
        if self.risk_manager.check_max_drawdown():
            self.alerts.error_alert(f"⛔ MAX DRAWDOWN EXCEEDED! Stopping bot.")
            self.running = False
            return
        
        # Run strategies
        #commented out other strategies because they aren't being used cause they don't perform well
        #self.run_4h_strategies()
        #self.run_hydra_strategy()
        self.run_grid_strategy()
        #self.run_vwap_strategy() 
        #self.run_momentum_strategy() 
        
    def run(self, interval_seconds: int = None):
        """
        Run the trading bot continuously.
        
        Args:
            interval_seconds: Override check interval (default: 60 seconds for HYDRA compatibility)
        """
        self.running = True
        
        mode_str = "PAPER TRADING" if self.paper_mode else "LIVE TRADING"
        self.alerts.status_alert(f"🚀 Bot started - {mode_str}")
        self.alerts.status_alert(f"Monitoring: {', '.join(self.assets)}")
        self.alerts.status_alert(f"Strategies: {', '.join(s.upper() for s in self.enabled_strategies)}")
        
        print(f"\n{'='*70}")
        print(f"🤖 MULTI-STRATEGY TRADING BOT - {mode_str}")
        print(f"{'='*70}")
        print(f"   Exchange: {self.exchange_id}")
        print(f"   Assets: {', '.join(self.assets)}")
        print(f"   Strategies: {', '.join(s.upper() for s in self.enabled_strategies)}")
        print(f"   Timeframes: 15m (Grid adaptive)")
        print(f"   Capital: ${self.capital:,.2f}")
        print(f"   Risk/Trade: {self.risk_per_trade*100:.1f}%")
        print(f"\n   Press Ctrl+C to stop\n")
        
        # Default to 60 second interval for HYDRA compatibility
        # HYDRA needs frequent checks on 15m timeframe
        if interval_seconds is None:
            interval_seconds = 60  # 1 minute default
        
        try:
            while self.running:
                # Run trading logic
                self.run_once()
                
                # Print status
                self.print_status()
                
                # # Calculate wait time
                # # For HYDRA, we need to check more frequently
                # if 'hydra' in self.enabled_strategies:
                #     # Check every interval_seconds
                #     wait_time = interval_seconds
                # else:
                #     # For 4h strategies only, can wait longer
                #     next_candle = self.get_next_candle_time('4h')
                #     wait_time = (next_candle - datetime.now()).total_seconds()
                #     wait_time = max(60, min(wait_time, 3600))  # Between 1 min and 1 hour
                
                # Check every interval_seconds
                wait_time = interval_seconds
                
                print(f"\n⏳ Next check in {wait_time/60:.1f} minutes...")
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n\n⛔ Bot stopped by user")
            self.alerts.status_alert("🛑 Bot stopped")
            for grid in self.grids.values():
                grid.shutdown()
            self._save_state()
        except Exception as e:
            self.alerts.error_alert(f"Bot crashed: {e}")
            traceback.print_exc()
            for grid in self.grids.values():
                grid.shutdown()
            self._save_state()
            raise


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Strategy Live Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  breakout  - 4h consolidation breakout with BB squeeze detection
  mean_rev  - 4h oversold bounce (RSI + Bollinger Band)
  hydra     - 15m regime-gated multi-engine strategy

Examples:
  python live_trader.py                              # All strategies, paper mode
  python live_trader.py --live                       # All strategies, live trading
  python live_trader.py --strategies breakout hydra  # Specific strategies
  python live_trader.py --strategies hydra           # HYDRA only
  python live_trader.py --assets SOL/USD ETH/USD     # Specific assets
  python live_trader.py --interval 300               # Check every 5 minutes
        """
    )
    
    parser.add_argument(
        '--assets', '-a',
        nargs='+',
        default=None,
        help='Assets to trade (e.g., LINK/USD SOL/USD)'
    )
    
    parser.add_argument(
        '--strategies', '-s',
        nargs='+',
        default=None,
        choices=['breakout', 'mean_rev', 'hydra', 'grid', 'vwap', 'momentum'],
        help='Strategies to run (default: all)'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live trading (default: paper trading)'
    )
    
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=STARTING_CAPITAL,
        help=f'Starting capital (default: {STARTING_CAPITAL})'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=None,
        help='Check interval in seconds (default: 60 for HYDRA, candle close for 4h only)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run one iteration and exit'
    )
    
    args = parser.parse_args()
    
    # Create trader
    trader = LiveTrader(
        assets=args.assets,
        strategies=args.strategies,
        capital=args.capital,
        paper_mode=not args.live
    )
    
    if args.test:
        trader.run_once()
        trader.print_status()
    else:
        trader.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
