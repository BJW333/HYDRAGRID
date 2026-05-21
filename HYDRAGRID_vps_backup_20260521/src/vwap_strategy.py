"""
VWAP Mean Reversion Strategy
==============================
Fades extreme deviations from the Volume-Weighted Average Price.

How it works:
    1. Calculate rolling VWAP and standard deviation bands
    2. When price hits -2σ band + RSI < 30 → BUY, target VWAP
    3. When price hits +2σ band + RSI > 70 → SELL, target VWAP
    4. Stop-loss just beyond ±3σ band
    5. Typical R:R ~2:1, win rate ~55-65%

Runs on 5-minute candles for responsive signals.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VWAPConfig:
    """VWAP reversion parameters."""
    # Signal thresholds
    entry_std_devs: float = 2.5      # was 2.0 — wait for more extreme move
    stop_std_devs: float = 4.0       # was 3.0 — crypto 3σ moves are normal
    rsi_oversold: float = 25         # was 30 — require more extreme RSI
    rsi_overbought: float = 75       # was 70 — same
    rsi_period: int = 14
    
    # VWAP calculation
    vwap_lookback: int = 72          # was 48 — 6hr lookback for stabler VWAP
    
    # Risk
    trade_size_pct: float = 0.08
    max_trades_per_day: int = 4      # was 6 — fewer but better trades
    cooldown_bars: int = 12          # was 6 — 1hr cooldown, less overtrading
    
    # Exit
    take_profit_at_vwap: bool = True
    partial_exit_pct: float = 0.5
    max_hold_bars: int = 24          # was 36 — cut dead trades at 2hrs not 3hrs


class VWAPStrategy:
    """
    VWAP Mean Reversion engine.
    
    Designed for 5-minute candles on crypto pairs.
    Plugs into LiveTrader's main loop.
    """
    
    def __init__(self, config: VWAPConfig = None, capital: float = 1000, alerts=None):
        self.config = config or VWAPConfig()
        self.capital = capital
        self.alerts = alerts
        
        # Active trade tracking
        self.active_trade: Optional[Dict] = None
        self.trades_today: int = 0
        self.trade_date: str = ''
        self.bars_since_trade: int = 999  # Cooldown counter
        
        # Performance
        self.completed_trades: list = []
        self.total_profit: float = 0
        
        # State
        self.state_file = Path(__file__).parent.parent / "data" / "vwap_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP, standard deviation bands, and RSI on 5min data.
        Expects df with columns: open, high, low, close, volume
        """
        df = df.copy()
        n = self.config.vwap_lookback
        
        # VWAP = cumulative(price × volume) / cumulative(volume)
        # Use rolling window instead of session-based (crypto has no sessions)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_vol = typical_price * df['volume']
        
        df['vwap'] = tp_vol.rolling(n).sum() / df['volume'].rolling(n).sum()
        
        # Standard deviation of price around VWAP
        df['vwap_std'] = (df['close'] - df['vwap']).rolling(n).std()
        
        # Bands
        df['vwap_upper_2'] = df['vwap'] + (self.config.entry_std_devs * df['vwap_std'])
        df['vwap_lower_2'] = df['vwap'] - (self.config.entry_std_devs * df['vwap_std'])
        df['vwap_upper_3'] = df['vwap'] + (self.config.stop_std_devs * df['vwap_std'])
        df['vwap_lower_3'] = df['vwap'] - (self.config.stop_std_devs * df['vwap_std'])
        
        # Z-score (how many std devs from VWAP)
        df['vwap_z'] = (df['close'] - df['vwap']) / (df['vwap_std'] + 1e-10)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(span=self.config.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.config.rsi_period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def check_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        Check for VWAP reversion entry signal.
        
        Args:
            df: DataFrame with 5min OHLCV data (needs ~60+ bars)
            symbol: Trading pair
            
        Returns:
            Signal dict or None
        """
        if len(df) < self.config.vwap_lookback + 10:
            return None
        
        # Daily trade limit
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.trade_date:
            self.trade_date = today
            self.trades_today = 0
        
        if self.trades_today >= self.config.max_trades_per_day:
            return None
        
        # Cooldown
        if self.bars_since_trade < self.config.cooldown_bars:
            self.bars_since_trade += 1
            return None
        
        # Skip if already in a trade
        if self.active_trade is not None:
            return None
        
        # Calculate features
        df = self.calculate_features(df)
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check for valid data
        if pd.isna(current['vwap']) or pd.isna(current['rsi']):
            return None
        
        price = current['close']
        vwap = current['vwap']
        z = current['vwap_z']
        rsi = current['rsi']
        
        signal = None
        
        # LONG: price at or below -2σ, RSI oversold, starting to recover
        if (z <= -self.config.entry_std_devs and 
            rsi < self.config.rsi_oversold and
            current['close'] > prev['close']):  # Price turning up
            
            stop = current['vwap_lower_3']
            target = vwap  # Target the mean
            risk = price - stop
            reward = target - price
            
            if risk > 0 and reward / risk >= 1.5:  # Minimum 1.5:1 R:R
                signal = {
                    'direction': 'LONG',
                    'entry': price,
                    'stop': stop,
                    'target': target,
                    'rsi': rsi,
                    'z_score': z,
                    'vwap': vwap,
                    'risk_reward': reward / risk,
                }
        
        # SHORT: price at or above +2σ, RSI overbought, starting to drop
        elif (z >= self.config.entry_std_devs and
              rsi > self.config.rsi_overbought and
              current['close'] < prev['close']):  # Price turning down
            
            stop = current['vwap_upper_3']
            target = vwap
            risk = stop - price
            reward = price - target
            
            if risk > 0 and reward / risk >= 1.5:
                signal = {
                    'direction': 'SHORT',
                    'entry': price,
                    'stop': stop,
                    'target': target,
                    'rsi': rsi,
                    'z_score': z,
                    'vwap': vwap,
                    'risk_reward': reward / risk,
                }
        
        return signal
    
    def open_trade(self, signal: Dict, symbol: str):
        """Open a VWAP reversion trade."""
        trade_size_usd = self.capital * self.config.trade_size_pct
        qty = trade_size_usd / signal['entry']
        
        self.active_trade = {
            'symbol': symbol,
            'direction': signal['direction'],
            'entry_price': signal['entry'],
            'stop_loss': signal['stop'],
            'take_profit': signal['target'],
            'qty': qty,
            'usd_size': trade_size_usd,
            'entry_time': datetime.now().isoformat(),
            'bars_held': 0,
            'rsi_at_entry': signal['rsi'],
            'z_at_entry': signal['z_score'],
            'risk_reward': signal['risk_reward'],
        }
        
        self.trades_today += 1
        self.bars_since_trade = 0
        
        msg = (
            f"🔮 VWAP {signal['direction']} {symbol} @ ${signal['entry']:,.2f}\n"
            f"Z-score: {signal['z_score']:.2f} | RSI: {signal['rsi']:.0f}\n"
            f"Stop: ${signal['stop']:,.2f} | Target: ${signal['target']:,.2f} "
            f"(R:R {signal['risk_reward']:.1f}:1)"
        )
        logger.info(msg)
        if self.alerts:
            self.alerts.send(msg, level="SIGNAL")
        
        self._save_state()
    
    def check_exit(self, current_price: float, current_bar: int = 0) -> Optional[str]:
        """
        Check if active trade should be closed.
        
        Returns exit reason or None.
        """
        if self.active_trade is None:
            return None
        
        trade = self.active_trade
        trade['bars_held'] += 1
        
        if trade['direction'] == 'LONG':
            if current_price <= trade['stop_loss']:
                return 'STOP'
            if current_price >= trade['take_profit']:
                return 'TARGET'
        else:  # SHORT
            if current_price >= trade['stop_loss']:
                return 'STOP'
            if current_price <= trade['take_profit']:
                return 'TARGET'
        
        # Time-based exit
        if trade['bars_held'] >= self.config.max_hold_bars:
            return 'TIMEOUT'
        
        return None
    
    def close_trade(self, exit_price: float, reason: str):
        """Close the active trade and record P&L."""
        if self.active_trade is None:
            return
        
        trade = self.active_trade
        
        if trade['direction'] == 'LONG':
            pnl = (exit_price - trade['entry_price']) * trade['qty']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['qty']
        
        # Subtract fees
        fee = (trade['entry_price'] * trade['qty'] * 0.0015) + \
              (exit_price * trade['qty'] * 0.0015)
        pnl -= fee
        
        self.total_profit += pnl
        
        record = {
            'symbol': trade['symbol'],
            'direction': trade['direction'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'qty': trade['qty'],
            'pnl': pnl,
            'reason': reason,
            'bars_held': trade['bars_held'],
            'z_at_entry': trade['z_at_entry'],
            'rsi_at_entry': trade['rsi_at_entry'],
            'time': datetime.now().isoformat(),
        }
        self.completed_trades.append(record)
        
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} VWAP CLOSED {trade['symbol']} | {reason}\n"
            f"P&L: ${pnl:,.2f} | Total: ${self.total_profit:,.2f} "
            f"({len(self.completed_trades)} trades)"
        )
        logger.info(msg)
        if self.alerts:
            self.alerts.send(msg, level="TRADE")
        
        self.active_trade = None
        self.bars_since_trade = 0
        self._save_state()
    
    def get_status(self) -> dict:
        """Get current strategy status."""
        wins = len([t for t in self.completed_trades if t['pnl'] > 0])
        total = len(self.completed_trades)
        return {
            'active_trade': self.active_trade is not None,
            'direction': self.active_trade['direction'] if self.active_trade else None,
            'symbol': self.active_trade['symbol'] if self.active_trade else None,
            'completed': total,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'profit': self.total_profit,
            'trades_today': self.trades_today,
        }
    
    def _save_state(self):
        """Persist state."""
        state = {
            'active_trade': self.active_trade,
            'total_profit': self.total_profit,
            'trades_today': self.trades_today,
            'trade_date': self.trade_date,
            'completed_trades': self.completed_trades[-50:],  # Keep last 50
            'last_update': datetime.now().isoformat(),
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> bool:
        """Load persisted state."""
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.active_trade = state.get('active_trade')
            self.total_profit = state.get('total_profit', 0)
            self.trades_today = state.get('trades_today', 0)
            self.trade_date = state.get('trade_date', '')
            self.completed_trades = state.get('completed_trades', [])
            return True
        except Exception as e:
            logger.error(f"Failed to load VWAP state: {e}")
            return False


def backtest_vwap(df_5m: pd.DataFrame, capital: float = 1000, 
                  config: VWAPConfig = None) -> Dict:
    """
    Backtest VWAP reversion on historical 5-minute data.
    
    Args:
        df_5m: DataFrame with 5min OHLCV (index=timestamp)
        capital: Starting capital
        config: Strategy config
        
    Returns:
        Dict with performance metrics and trade list
    """
    cfg = config or VWAPConfig()
    strategy = VWAPStrategy(config=cfg, capital=capital)
    
    # Calculate features on full dataset
    df = strategy.calculate_features(df_5m)
    
    trades = []
    equity = capital
    peak_equity = capital
    max_dd = 0
    
    for i in range(cfg.vwap_lookback + 10, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        
        # Check exit first
        if strategy.active_trade:
            strategy.active_trade['bars_held'] += 1
            trade = strategy.active_trade
            
            exit_reason = None
            if trade['direction'] == 'LONG':
                if price <= trade['stop_loss']:
                    exit_reason = 'STOP'
                elif price >= trade['take_profit']:
                    exit_reason = 'TARGET'
            else:
                if price >= trade['stop_loss']:
                    exit_reason = 'STOP'
                elif price <= trade['take_profit']:
                    exit_reason = 'TARGET'
            
            if trade['bars_held'] >= cfg.max_hold_bars:
                exit_reason = 'TIMEOUT'
            
            if exit_reason:
                if trade['direction'] == 'LONG':
                    pnl = (price - trade['entry_price']) * trade['qty']
                else:
                    pnl = (trade['entry_price'] - price) * trade['qty']
                
                fee = (trade['entry_price'] * trade['qty'] + price * trade['qty']) * 0.0015
                pnl -= fee
                equity += pnl
                
                trades.append({
                    'entry': trade['entry_price'],
                    'exit': price,
                    'direction': trade['direction'],
                    'pnl': pnl,
                    'reason': exit_reason,
                    'bars': trade['bars_held'],
                    'z': trade.get('z_at_entry', 0),
                })
                
                strategy.active_trade = None
                strategy.bars_since_trade = 0
                strategy.capital = equity
                
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity
                max_dd = max(max_dd, dd)
            
            continue
        
        # Cooldown
        strategy.bars_since_trade += 1
        if strategy.bars_since_trade < cfg.cooldown_bars:
            continue
        
        # Check for signal
        if pd.isna(row['vwap']) or pd.isna(row['rsi']):
            continue
        
        z = row['vwap_z']
        rsi = row['rsi']
        vwap = row['vwap']
        
        signal = None
        
        # Long signal
        if (z <= -cfg.entry_std_devs and rsi < cfg.rsi_oversold and 
            row['close'] > prev['close']):
            stop = row['vwap_lower_3']
            target = vwap
            risk = price - stop
            reward = target - price
            if risk > 0 and reward / risk >= 1.5:
                signal = {'direction': 'LONG', 'stop': stop, 'target': target}
        
        # Short signal
        elif (z >= cfg.entry_std_devs and rsi > cfg.rsi_overbought and
              row['close'] < prev['close']):
            stop = row['vwap_upper_3']
            target = vwap
            risk = stop - price
            reward = price - target
            if risk > 0 and reward / risk >= 1.5:
                signal = {'direction': 'SHORT', 'stop': stop, 'target': target}
        
        if signal:
            trade_size = equity * cfg.trade_size_pct
            qty = trade_size / price
            strategy.active_trade = {
                'direction': signal['direction'],
                'entry_price': price,
                'stop_loss': signal['stop'],
                'take_profit': signal['target'],
                'qty': qty,
                'bars_held': 0,
                'z_at_entry': z,
                'rsi_at_entry': rsi,
            }
            strategy.bars_since_trade = 0
    
    # Metrics
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    return {
        'total_trades': len(trades),
        'winners': len(wins),
        'losers': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'total_pnl': sum(t['pnl'] for t in trades),
        'avg_win': sum(t['pnl'] for t in wins) / len(wins) if wins else 0,
        'avg_loss': sum(t['pnl'] for t in losses) / len(losses) if losses else 0,
        'max_drawdown': max_dd * 100,
        'final_equity': equity,
        'return_pct': (equity / capital - 1) * 100,
        'trades': trades,
        'by_reason': {
            r: len([t for t in trades if t['reason'] == r])
            for r in ['TARGET', 'STOP', 'TIMEOUT']
        }
    }
