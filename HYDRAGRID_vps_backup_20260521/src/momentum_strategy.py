"""
Momentum Breakout Trailer
===========================
Catches trending moves that Grid can't capture.

How it works:
    1. Monitor hourly price change on 5min candles
    2. When price moves >1.5% in 1 hour with above-avg volume → enter
    3. Trail stop at 1.5% below the peak price
    4. Ride until trail catches you — small losses, occasional big wins
    
Complements Grid: Grid profits from ranging, Momentum profits from trending.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MomentumConfig:
    """Momentum trailer parameters."""
    trigger_pct: float = 0.015        # 1.5% move in lookback triggers entry
    lookback_bars: int = 12           # 12 × 5min = 1 hour lookback
    trail_pct: float = 0.015          # 1.5% trailing stop from peak
    volume_threshold: float = 1.3     # Volume must be 30% above average
    volume_avg_period: int = 48       # 4hr volume average
    
    trade_size_pct: float = 0.08      # 8% of capital per trade
    max_trades_per_day: int = 5       # Cap daily trades
    cooldown_bars: int = 6            # 30min cooldown between trades
    max_hold_bars: int = 72           # 6hr max hold (72 × 5min)


class MomentumTrailer:
    """
    Momentum Breakout Trailing strategy.
    
    Enters when price surges, rides with a trailing stop.
    Designed for 5-minute candles.
    """
    
    def __init__(self, config: MomentumConfig = None, capital: float = 1000,
                 alerts=None):
        self.config = config or MomentumConfig()
        self.capital = capital
        self.alerts = alerts
        
        # Active trade
        self.active_trade: Optional[Dict] = None
        self.peak_price: float = 0
        
        # Daily tracking
        self.trades_today: int = 0
        self.trade_date: str = ''
        self.bars_since_trade: int = 999
        
        # Performance
        self.completed_trades: list = []
        self.total_profit: float = 0
        
        # State
        self.state_file = Path(__file__).parent.parent / "data" / "momentum_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    def check_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        Check for momentum breakout signal.
        
        Args:
            df: DataFrame with 5min OHLCV (needs 60+ bars)
            symbol: Trading pair
            
        Returns:
            Signal dict or None
        """
        cfg = self.config
        
        if len(df) < cfg.volume_avg_period + cfg.lookback_bars:
            return None
        
        # Daily limit
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.trade_date:
            self.trade_date = today
            self.trades_today = 0
        
        if self.trades_today >= cfg.max_trades_per_day:
            return None
        
        # Cooldown
        self.bars_since_trade += 1
        if self.bars_since_trade < cfg.cooldown_bars:
            return None
        
        # Skip if already in a trade
        if self.active_trade is not None:
            return None
        
        current = df.iloc[-1]
        price_now = current['close']
        
        # Calculate price change over lookback period
        price_then = df.iloc[-cfg.lookback_bars]['close']
        pct_change = (price_now - price_then) / price_then
        
        # Check volume
        avg_volume = df['volume'].iloc[-cfg.volume_avg_period:].mean()
        recent_volume = df['volume'].iloc[-cfg.lookback_bars:].sum()
        expected_volume = avg_volume * cfg.lookback_bars
        volume_ratio = recent_volume / expected_volume if expected_volume > 0 else 0
        
        # LONG signal: price up 1.5%+ with strong volume
        if (pct_change >= cfg.trigger_pct and 
            volume_ratio >= cfg.volume_threshold):
            
            return {
                'direction': 'LONG',
                'entry': price_now,
                'pct_change': pct_change,
                'volume_ratio': volume_ratio,
                'trail_price': price_now * (1 - cfg.trail_pct),
            }
        
        return None
    
    def open_trade(self, signal: Dict, symbol: str):
        """Open a momentum trade."""
        trade_size_usd = self.capital * self.config.trade_size_pct
        qty = trade_size_usd / signal['entry']
        
        self.active_trade = {
            'symbol': symbol,
            'direction': signal['direction'],
            'entry_price': signal['entry'],
            'qty': qty,
            'usd_size': trade_size_usd,
            'entry_time': datetime.now().isoformat(),
            'bars_held': 0,
            'trigger_pct': signal['pct_change'],
            'volume_ratio': signal['volume_ratio'],
        }
        self.peak_price = signal['entry']
        
        self.trades_today += 1
        self.bars_since_trade = 0
        
        trail = signal['entry'] * (1 - self.config.trail_pct)
        msg = (
            f"🚀 MOMENTUM LONG {symbol} @ ${signal['entry']:,.2f}\n"
            f"Move: +{signal['pct_change']*100:.1f}% | Vol: {signal['volume_ratio']:.1f}x\n"
            f"Trail stop: ${trail:,.2f} (-{self.config.trail_pct*100:.1f}%)"
        )
        logger.info(msg)
        if self.alerts:
            self.alerts.send(msg, level="SIGNAL")
        
        self._save_state()
    
    def update_and_check_exit(self, current_price: float) -> Optional[str]:
        """
        Update trailing stop and check for exit.
        Call every iteration with current price.
        
        Returns exit reason or None.
        """
        if self.active_trade is None:
            return None
        
        self.active_trade['bars_held'] += 1
        
        # Update peak
        if current_price > self.peak_price:
            self.peak_price = current_price
        
        # Calculate trail stop (always 1.5% below peak)
        trail_stop = self.peak_price * (1 - self.config.trail_pct)
        
        # Check trail stop
        if current_price <= trail_stop:
            return 'TRAIL'
        
        # Time-based exit
        if self.active_trade['bars_held'] >= self.config.max_hold_bars:
            return 'TIMEOUT'
        
        return None
    
    def close_trade(self, exit_price: float, reason: str):
        """Close active trade and record P&L."""
        if self.active_trade is None:
            return
        
        trade = self.active_trade
        
        pnl = (exit_price - trade['entry_price']) * trade['qty']
        
        # Fees
        fee = (trade['entry_price'] * trade['qty'] * 0.0015) + \
              (exit_price * trade['qty'] * 0.0015)
        pnl -= fee
        
        self.total_profit += pnl
        
        pct_gain = (exit_price / trade['entry_price'] - 1) * 100
        peak_pct = (self.peak_price / trade['entry_price'] - 1) * 100
        
        record = {
            'symbol': trade['symbol'],
            'direction': trade['direction'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'peak_price': self.peak_price,
            'qty': trade['qty'],
            'pnl': pnl,
            'pct': pct_gain,
            'peak_pct': peak_pct,
            'reason': reason,
            'bars_held': trade['bars_held'],
            'time': datetime.now().isoformat(),
        }
        self.completed_trades.append(record)
        
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} MOMENTUM CLOSED {trade['symbol']} | {reason}\n"
            f"Entry: ${trade['entry_price']:,.2f} → Exit: ${exit_price:,.2f} "
            f"({pct_gain:+.1f}%)\n"
            f"Peak: ${self.peak_price:,.2f} ({peak_pct:+.1f}%) | "
            f"P&L: ${pnl:,.2f} | Total: ${self.total_profit:,.2f}"
        )
        logger.info(msg)
        if self.alerts:
            self.alerts.send(msg, level="TRADE")
        
        self.active_trade = None
        self.peak_price = 0
        self.bars_since_trade = 0
        self._save_state()
    
    def get_status(self) -> dict:
        """Get current status."""
        wins = len([t for t in self.completed_trades if t['pnl'] > 0])
        total = len(self.completed_trades)
        
        status = {
            'active_trade': self.active_trade is not None,
            'symbol': self.active_trade['symbol'] if self.active_trade else None,
            'completed': total,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'profit': self.total_profit,
            'trades_today': self.trades_today,
        }
        
        if self.active_trade:
            status['direction'] = self.active_trade['direction']
            status['entry'] = self.active_trade['entry_price']
            status['peak'] = self.peak_price
            status['trail'] = self.peak_price * (1 - self.config.trail_pct)
        
        return status
    
    def _save_state(self):
        """Persist state."""
        state = {
            'active_trade': self.active_trade,
            'peak_price': self.peak_price,
            'total_profit': self.total_profit,
            'trades_today': self.trades_today,
            'trade_date': self.trade_date,
            'completed_trades': self.completed_trades[-50:],
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
            self.peak_price = state.get('peak_price', 0)
            self.total_profit = state.get('total_profit', 0)
            self.trades_today = state.get('trades_today', 0)
            self.trade_date = state.get('trade_date', '')
            self.completed_trades = state.get('completed_trades', [])
            return True
        except Exception as e:
            logger.error(f"Failed to load momentum state: {e}")
            return False


def backtest_momentum(df_5m: pd.DataFrame, capital: float = 1000,
                      config: MomentumConfig = None) -> Dict:
    """
    Backtest momentum trailer on historical 5-minute data.
    
    Returns dict with performance metrics.
    """
    cfg = config or MomentumConfig()
    
    if len(df_5m) < cfg.volume_avg_period + cfg.lookback_bars + 10:
        return {'total_trades': 0, 'total_pnl': 0, 'trades': []}
    
    trades = []
    equity = capital
    peak_equity = capital
    max_dd = 0
    
    active = None
    peak_price = 0
    bars_since = 999
    cooldown = cfg.cooldown_bars
    
    # Pre-calculate volume average
    vol_avg = df_5m['volume'].rolling(cfg.volume_avg_period).mean()
    
    for i in range(cfg.volume_avg_period + cfg.lookback_bars, len(df_5m)):
        price = df_5m.iloc[i]['close']
        
        # Exit check
        if active:
            active['bars'] += 1
            if price > peak_price:
                peak_price = price
            
            trail = peak_price * (1 - cfg.trail_pct)
            reason = None
            
            if price <= trail:
                reason = 'TRAIL'
            elif active['bars'] >= cfg.max_hold_bars:
                reason = 'TIMEOUT'
            
            if reason:
                pnl = (price - active['entry']) * active['qty']
                fee = (active['entry'] + price) * active['qty'] * 0.0015
                pnl -= fee
                equity += pnl
                
                trades.append({
                    'entry': active['entry'],
                    'exit': price,
                    'peak': peak_price,
                    'pnl': pnl,
                    'pct': (price / active['entry'] - 1) * 100,
                    'reason': reason,
                    'bars': active['bars'],
                })
                
                active = None
                peak_price = 0
                bars_since = 0
                
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                max_dd = max(max_dd, dd)
            continue
        
        # Cooldown
        bars_since += 1
        if bars_since < cooldown:
            continue
        
        # Signal check
        price_then = df_5m.iloc[i - cfg.lookback_bars]['close']
        pct_change = (price - price_then) / price_then
        
        recent_vol = df_5m['volume'].iloc[i - cfg.lookback_bars:i].sum()
        expected_vol = vol_avg.iloc[i] * cfg.lookback_bars if not pd.isna(vol_avg.iloc[i]) else 0
        vol_ratio = recent_vol / expected_vol if expected_vol > 0 else 0
        
        if pct_change >= cfg.trigger_pct and vol_ratio >= cfg.volume_threshold:
            trade_size = equity * cfg.trade_size_pct
            qty = trade_size / price
            active = {'entry': price, 'qty': qty, 'bars': 0}
            peak_price = price
            bars_since = 0
    
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
            for r in ['TRAIL', 'TIMEOUT']
        },
        'avg_peak_pct': np.mean([t['pct'] for t in wins]) if wins else 0,
    }
