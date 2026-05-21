#!/usr/bin/env python3
"""
Strategy Suite Backtester
==========================
Backtest all active strategies: HYDRA, Grid, and VWAP.

Usage:
    python3.10 backtest.py                              # 90 days, $1k, all strategies
    python3.10 backtest.py --days 365 --capital 5000    # Full year at $5k
    python3.10 backtest.py --strategy hydra              # HYDRA only
    python3.10 backtest.py --strategy grid               # Grid only
    python3.10 backtest.py --strategy vwap               # VWAP only
    python3.10 backtest.py --symbol SOL/USD              # Single asset
    python3.10 backtest.py --debug                       # Show HYDRA debug info

Strategies:
    HYDRA: 15-minute regime-gated multi-engine directional trades
    GRID:  Limit order grid capturing oscillation profit
    VWAP:  5-minute mean reversion to volume-weighted average price
"""

import sys
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    STARTING_CAPITAL, EXCHANGE, ASSET_CONFIGS,
    COMMISSION, SLIPPAGE
)
from src.data_fetcher import DataFetcher
from src.hydra_spot import (
    HydraSpotStrategy, HydraConfig, FeatureEngine,
    RegimeDetector, AlphaEngines
)
from src.vwap_strategy import backtest_vwap, VWAPConfig
from src.momentum_strategy import backtest_momentum, MomentumConfig

DEFAULT_ASSETS = ['SOL/USD', 'ETH/USD', 'LINK/USD', 'AVAX/USD']
HYDRA_TIMEFRAME = '15m'


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class StrategyResult:
    """Results from a single strategy backtest."""
    strategy: str
    symbol: str
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: float
    final_equity: float
    return_pct: float
    max_drawdown: float
    days_tested: int = 0
    sharpe_ratio: float = 0.0
    monthly_return: float = 0.0
    trades_per_week: float = 0.0
    exit_reasons: Dict = field(default_factory=dict)
    by_engine: Dict = field(default_factory=dict)
    debug_info: Dict = field(default_factory=dict)


# =============================================================================
# DATA FETCHING
# =============================================================================

# def fetch_data(symbol: str, timeframe: str, days: int = 90) -> Optional[pd.DataFrame]:
#     """Fetch historical data from exchange."""
#     try:
#         fetcher = DataFetcher(exchange_id=EXCHANGE)
        
#         if timeframe in ('5m', '15m'):
#             # Short timeframes: use fetch_live with candle count
#             candles_per_day = {'5m': 288, '15m': 96}[timeframe]
#             num_candles = min(days * candles_per_day, 10000)
#             df = fetcher.fetch_live(symbol, timeframe=timeframe, num_candles=num_candles)
#         else:
#             start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
#             end = datetime.now().strftime('%Y-%m-%d')
#             df = fetcher.fetch_historical(symbol, timeframe, start, end)
        
#         if df is None or len(df) == 0:
#             print(f"      ❌ No data returned")
#             return None
        
#         return df
#     except Exception as e:
#         print(f"      ❌ Error fetching {symbol} {timeframe}: {e}")
#         return None

def fetch_data(symbol: str, timeframe: str, days: int = 90) -> Optional[pd.DataFrame]:
    """Fetch data — tries local CSV first, then API."""
    # Check for downloaded CSV first (from download_data.py)
    # Binance data uses USDT pairs, map from USD
    csv_symbol = symbol
    #csv_symbol = symbol.replace('/USD', '/USDT')

    clean = csv_symbol.replace('/', '_')
    csv_path = Path(__file__).parent / "data" / "historical" / f"{clean}_{timeframe}.csv"
    
    if csv_path.exists():
        print(f"   Loading from {csv_path.name}...")
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        # Filter to requested days
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff]
        print(f"   Got {len(df)} candles ({(df.index[-1] - df.index[0]).days} days)")
        return df if len(df) > 0 else None
    
    # Fall back to live API
    try:
        fetcher = DataFetcher(exchange_id=EXCHANGE)
        
        if timeframe in ('5m', '15m'):
            candles_per_day = {'5m': 288, '15m': 96}[timeframe]
            num_candles = min(days * candles_per_day, 10000)
            df = fetcher.fetch_live(symbol, timeframe=timeframe, num_candles=num_candles)
        else:
            start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            end = datetime.now().strftime('%Y-%m-%d')
            df = fetcher.fetch_historical(symbol, timeframe, start, end)
        
        if df is None or len(df) == 0:
            print(f"      ❌ No data returned")
            return None
        
        return df
    except Exception as e:
        print(f"      ❌ Error fetching {symbol} {timeframe}: {e}")
        return None

# =============================================================================
# HYDRA BACKTESTER (uses real hydra_spot.py engine)
# =============================================================================

class HydraBacktester:
    """Backtest the HYDRA-SPOT strategy using the actual signal engine."""
    
    def __init__(self, df: pd.DataFrame, cfg: HydraConfig = None,
                 symbol: str = 'TEST', starting_capital: float = 1000,
                 debug: bool = False):
        self.df = df.copy()
        self.cfg = cfg or HydraConfig()
        self.symbol = symbol
        self.starting_capital = starting_capital
        self.debug = debug
        
        self.strategy = HydraSpotStrategy(self.cfg)
        self.strategy.equity = starting_capital
        self.strategy.starting_equity = starting_capital
        self.strategy.peak_equity = starting_capital
        self.strategy.day_start_equity = starting_capital
        
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        
        self.debug_stats = {
            'bars_processed': 0,
            'regime_failures': 0,
            'no_candidates': 0,
            'quality_filtered': 0,
            'signals_generated': 0,
            'regime_samples': [],
            'candidate_samples': [],
        }
    
    def run(self) -> StrategyResult:
        """Run the backtest."""
        print(f"      Running HYDRA on {len(self.df)} bars...")
        
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'time' in self.df.columns:
                self.df['time'] = pd.to_datetime(self.df['time'])
                self.df.set_index('time', inplace=True)
            else:
                self.df.index = pd.to_datetime(self.df.index)
        
        warmup = 150
        peak_equity = self.strategy.equity
        max_dd = 0
        
        if self.debug and len(self.df) > warmup:
            features = FeatureEngine()
            regime_detector = RegimeDetector(self.cfg)
            engines = AlphaEngines(self.cfg)
        
        for i in range(warmup, len(self.df)):
            candles = self.df.iloc[:i+1].copy()
            
            if self.debug:
                self._debug_bar(candles, i, features, regime_detector, engines)
            
            actions = self.strategy.on_bar(self.symbol, candles, i)
            self.debug_stats['bars_processed'] += 1
            
            current_equity = self.strategy.equity
            self.equity_curve.append(current_equity)
            
            peak_equity = max(peak_equity, current_equity)
            dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            max_dd = max(max_dd, dd)
            self.drawdown_curve.append(dd)
            
            for action in actions:
                if action.get('action') == 'OPEN':
                    self.debug_stats['signals_generated'] += 1
                    self.trades.append({
                        'type': 'OPEN', 'bar': i, 'time': candles.index[-1],
                        'direction': action['direction'], 'engine': action['engine'],
                        'entry': action['entry'], 'quality': action['quality'],
                    })
                elif action.get('action') == 'CLOSE':
                    self.trades.append({
                        'type': 'CLOSE', 'bar': i, 'time': candles.index[-1],
                        'reason': action['reason'], 'price': action['price'],
                    })
        
        # Close any remaining positions
        if self.strategy.positions:
            final_price = float(self.df.iloc[-1]['close'])
            for pos in self.strategy.positions:
                if pos.active:
                    self.strategy._close_position(pos, final_price, 'END')
        
        return self._calculate_results(max_dd)
    
    def _debug_bar(self, candles, bar_idx, features, regime_detector, engines):
        if len(candles) < 100:
            return
        feats = features.compute(candles)
        if len(feats) < 100:
            return
        regime = regime_detector.detect(feats)
        max_conf = max(regime.values()) if regime else 0
        
        if bar_idx % 100 == 0 and len(self.debug_stats['regime_samples']) < 5:
            self.debug_stats['regime_samples'].append({
                'bar': bar_idx, 'regime': dict(regime), 'max_conf': max_conf
            })
        
        if max_conf < self.cfg.min_regime_confidence:
            self.debug_stats['regime_failures'] += 1
            return
        
        candidates = engines.generate(self.symbol, candles, feats, regime)
        if not candidates:
            self.debug_stats['no_candidates'] += 1
            return
        
        if len(self.debug_stats['candidate_samples']) < 5:
            self.debug_stats['candidate_samples'].append({
                'bar': bar_idx, 'num': len(candidates),
                'qualities': [c.quality_score for c in candidates],
                'engines': [c.engine for c in candidates],
            })
        
        qualified = [c for c in candidates if c.quality_score >= self.cfg.min_quality_score]
        if not qualified:
            self.debug_stats['quality_filtered'] += 1
    
    def _calculate_results(self, max_dd: float) -> StrategyResult:
        results = self.strategy.trade_results
        total = len(results)
        winners = sum(results) if results else 0
        losers = total - winners
        win_rate = (winners / total * 100) if total > 0 else 0
        
        final_equity = self.strategy.equity
        total_pnl = final_equity - self.strategy.starting_equity
        return_pct = (final_equity / self.strategy.starting_equity - 1) * 100
        
        days = (self.df.index[-1] - self.df.index[0]).days if len(self.df) > 1 else 0
        weeks = max(days / 7, 0.1)
        trades_per_week = total / weeks if weeks > 0 else 0
        monthly_return = (return_pct / days) * 30 if days > 0 else 0
        
        sharpe = 0
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        engine_trades = {}
        exit_reasons = {}
        for trade in self.trades:
            if trade['type'] == 'OPEN':
                eng = trade['engine']
                engine_trades[eng] = engine_trades.get(eng, {'count': 0})
                engine_trades[eng]['count'] += 1
            elif trade['type'] == 'CLOSE':
                reason = trade['reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return StrategyResult(
            strategy='HYDRA', symbol=self.symbol,
            total_trades=total, winners=winners, losers=losers,
            win_rate=win_rate, total_pnl=total_pnl,
            final_equity=final_equity, return_pct=return_pct,
            max_drawdown=max_dd * 100, days_tested=days,
            sharpe_ratio=round(sharpe, 2),
            trades_per_week=trades_per_week,
            monthly_return=round(monthly_return, 2),
            exit_reasons=exit_reasons, by_engine=engine_trades,
            debug_info=self.debug_stats,
        )


# =============================================================================
# GRID BACKTESTER
# =============================================================================

def backtest_grid(df: pd.DataFrame, capital: float, num_grids: int = 12,
                  range_pct: float = 0.08, alloc_pct: float = 0.85,
                  vol_weight: float = None, adaptive: bool = True) -> StrategyResult:
    """Simulate grid trading over historical data with adaptive range."""
    if vol_weight is not None:
        grid_capital = capital * alloc_pct * vol_weight
    else:
        grid_capital = capital * alloc_pct / 2  # Legacy: per pair
    
    symbol = 'GRID'
    
    if len(df) < 50:
        return StrategyResult(strategy='GRID', symbol=symbol, total_trades=0,
                              winners=0, losers=0, win_rate=0, total_pnl=0,
                              final_equity=capital, return_pct=0, max_drawdown=0)
    
    trades = []
    total_profit = 0
    equity = capital
    peak_equity = capital
    max_dd = 0
    
    # Candles per day for rebalance timing
    if len(df) > 1:
        td = (df.index[1] - df.index[0]).total_seconds()
        cpd = int(86400 / td) if td > 0 else 288
    else:
        cpd = 288
    
    rebalance = cpd * 7  # Weekly rebalance (or adaptive triggers sooner)
    buy_levels = num_grids // 2
    grid_center = 0
    grid_spacing = 0
    grid_buys = {}
    order_qty = 0
    current_range_pct = range_pct
    
    for i in range(1, len(df)):
        price = df.iloc[i]['close']
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        
        # Adaptive range: recalculate ATR every rebalance period
        needs_setup = (i % rebalance == 1 or grid_center == 0)
        
        # Also rebalance if price near grid edge (adaptive)
        if adaptive and grid_center > 0 and grid_spacing > 0:
            lower = grid_center * (1 - current_range_pct)
            upper = grid_center * (1 + current_range_pct)
            range_size = upper - lower
            if price < lower + range_size * 0.10 or price > upper - range_size * 0.10:
                needs_setup = True
        
        if needs_setup:
            grid_buys = {}
            grid_center = price
            
            # Adaptive: calculate ATR from recent data
            if adaptive and i > 96:
                lookback = min(96, i)
                recent = df.iloc[i-lookback:i]
                tr_vals = []
                for j in range(1, len(recent)):
                    h = recent.iloc[j]['high']
                    l = recent.iloc[j]['low']
                    c = recent.iloc[j-1]['close']
                    tr = max(h - l, abs(h - c), abs(l - c))
                    tr_vals.append(tr)
                if tr_vals:
                    atr = sum(tr_vals) / len(tr_vals)
                    atr_pct = atr / price
                    current_range_pct = max(0.03, min(0.15, atr_pct * 6))
            
            lower = price * (1 - current_range_pct)
            upper = price * (1 + current_range_pct)
            grid_spacing = (upper - lower) / num_grids
            if grid_spacing <= 0:
                continue
            usd_per_level = grid_capital / buy_levels
            order_qty = usd_per_level / price
            for g in range(1, buy_levels + 1):
                lp = round(price - (g * grid_spacing), 2)
                grid_buys[lp] = False
        
        if grid_spacing <= 0 or order_qty <= 0:
            continue
        
        # Check buy fills
        for lp, has_inv in list(grid_buys.items()):
            if not has_inv and low <= lp:
                grid_buys[lp] = True
        
        # Check sell fills
        for lp, has_inv in list(grid_buys.items()):
            if has_inv:
                sp = lp + grid_spacing
                if high >= sp:
                    profit = grid_spacing * order_qty
                    fee = (lp + sp) * order_qty * 0.0015
                    net = profit - fee
                    total_profit += net
                    equity += net
                    trades.append({'buy': lp, 'sell': sp, 'profit': net})
                    grid_buys[lp] = False
                    
                    peak_equity = max(peak_equity, equity)
                    dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    max_dd = max(max_dd, dd)
    
    days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 0
    avg_per_cycle = total_profit / len(trades) if trades else 0
    cycles_per_day = len(trades) / days if days > 0 else 0
    
    return StrategyResult(
        strategy='GRID', symbol=symbol,
        total_trades=len(trades), winners=len(trades), losers=0,
        win_rate=100.0 if trades else 0,
        total_pnl=total_profit, final_equity=equity,
        return_pct=(equity / capital - 1) * 100,
        max_drawdown=max_dd * 100, days_tested=days,
        exit_reasons={'avg_per_cycle': round(avg_per_cycle, 4),
                      'cycles_per_day': round(cycles_per_day, 1)},
    )


# =============================================================================
# DISPLAY
# =============================================================================

def print_result(r: StrategyResult):
    """Print formatted result for one strategy."""
    emoji = {'HYDRA': '🐉', 'GRID': '📐', 'VWAP': '🔮', 'MOMENTUM': '🚀'}.get(r.strategy, '📊')
    print(f"\n   {emoji} {r.strategy} — {r.symbol}")
    print(f"      Trades: {r.total_trades} | W: {r.winners} | L: {r.losers} | "
          f"WR: {r.win_rate:.0f}%")
    print(f"      P&L: ${r.total_pnl:,.2f} | Return: {r.return_pct:.1f}%")
    print(f"      Max DD: {r.max_drawdown:.1f}% | Days: {r.days_tested}")
    
    if r.strategy == 'HYDRA':
        if r.by_engine:
            engines = ', '.join(f"{e}: {d['count']}" for e, d in r.by_engine.items())
            print(f"      Engines: {engines}")
        if r.exit_reasons:
            exits = ', '.join(f"{k}: {v}" for k, v in r.exit_reasons.items())
            print(f"      Exits: {exits}")
        if r.sharpe_ratio:
            print(f"      Sharpe: {r.sharpe_ratio} | Trades/wk: {r.trades_per_week:.1f}")
    
    elif r.strategy == 'GRID':
        er = r.exit_reasons
        print(f"      Avg/cycle: ${er.get('avg_per_cycle', 0):.4f} | "
              f"Cycles/day: {er.get('cycles_per_day', 0):.1f}")
    
    elif r.strategy == 'VWAP':
        if r.exit_reasons:
            exits = ', '.join(f"{k}: {v}" for k, v in r.exit_reasons.items())
            print(f"      Exits: {exits}")
    
    if r.debug_info and r.debug_info.get('bars_processed'):
        d = r.debug_info
        print(f"      Debug: {d['bars_processed']} bars | "
              f"Signals: {d['signals_generated']} | "
              f"Regime fail: {d['regime_failures']} | "
              f"Quality filtered: {d['quality_filtered']}")


# =============================================================================
# MAIN
# =============================================================================

def run_backtest(symbols: List[str], strategies: List[str], days: int,
                 capital: float, debug: bool = False):
    """Run complete backtest."""
    
    print("=" * 65)
    print(f"STRATEGY SUITE BACKTEST")
    print(f"Capital: ${capital:,.0f} | Period: {days} days")
    print(f"Symbols: {', '.join(symbols)} | Strategies: {', '.join(strategies)}")
    print("=" * 65)
    
    all_results: List[StrategyResult] = []
    
    for symbol in symbols:
        print(f"\n{'='*65}")
        print(f"  {symbol}")
        print(f"{'='*65}")
        
        # --- HYDRA ---
        if 'hydra' in strategies:
            print(f"\n   Fetching 15m data for HYDRA...")
            df_15m = fetch_data(symbol, '15m', days)
            if df_15m is not None and len(df_15m) > 150:
                print(f"   Got {len(df_15m)} candles")
                bt = HydraBacktester(df_15m, symbol=symbol,
                                     starting_capital=capital, debug=debug)
                result = bt.run()
                all_results.append(result)
                print_result(result)
            else:
                print(f"   ❌ Not enough 15m data for HYDRA")
        
        # --- GRID ---
        if 'grid' in strategies:
            print(f"\n   Fetching 15m data for GRID...")
            if 'df_15m' not in dir() or df_15m is None:
                df_15m = fetch_data(symbol, '15m', days)
            if df_15m is not None and len(df_15m) > 50:
                grids = 10
                vol_weights = {'SOL/USD': 0.30, 'ETH/USD': 0.25, 'LINK/USD': 0.25, 'AVAX/USD': 0.20}
                vw = vol_weights.get(symbol, 0.33)
                result = backtest_grid(df_15m, capital, num_grids=grids,
                                       vol_weight=vw, adaptive=True)
                result.symbol = symbol
                all_results.append(result)
                print_result(result)
            else:
                print(f"   ❌ Not enough data for GRID")
        
        # --- VWAP ---
        if 'vwap' in strategies:
            print(f"\n   Fetching 5m data for VWAP...")
            df_5m = fetch_data(symbol, '5m', min(days, 59))
            if df_5m is not None and len(df_5m) > 60:
                print(f"   Got {len(df_5m)} candles")
                vwap_result = backtest_vwap(df_5m, capital)
                
                wins = [t for t in vwap_result['trades'] if t['pnl'] > 0]
                losses = [t for t in vwap_result['trades'] if t['pnl'] <= 0]
                d = (df_5m.index[-1] - df_5m.index[0]).days if len(df_5m) > 1 else 0
                
                result = StrategyResult(
                    strategy='VWAP', symbol=symbol,
                    total_trades=vwap_result['total_trades'],
                    winners=vwap_result['winners'],
                    losers=vwap_result['losers'],
                    win_rate=vwap_result['win_rate'],
                    total_pnl=vwap_result['total_pnl'],
                    final_equity=vwap_result['final_equity'],
                    return_pct=vwap_result['return_pct'],
                    max_drawdown=vwap_result['max_drawdown'],
                    days_tested=d,
                    exit_reasons=vwap_result['by_reason'],
                )
                all_results.append(result)
                print_result(result)
            else:
                print(f"   ❌ Not enough 5m data for VWAP (Alpaca free tier: ~59 days max)")

        # --- MOMENTUM ---
        if 'momentum' in strategies:
            print(f"\n   Fetching 5m data for MOMENTUM...")
            df_5m = fetch_data(symbol, '5m', days)
            if df_5m is not None and len(df_5m) > 60:
                print(f"   Got {len(df_5m)} candles")
                mom_result = backtest_momentum(df_5m, capital)
                
                d = (df_5m.index[-1] - df_5m.index[0]).days if len(df_5m) > 1 else 0
                
                result = StrategyResult(
                    strategy='MOMENTUM', symbol=symbol,
                    total_trades=mom_result['total_trades'],
                    winners=mom_result['winners'],
                    losers=mom_result['losers'],
                    win_rate=mom_result['win_rate'],
                    total_pnl=mom_result['total_pnl'],
                    final_equity=mom_result['final_equity'],
                    return_pct=mom_result['return_pct'],
                    max_drawdown=mom_result['max_drawdown'],
                    days_tested=d,
                    exit_reasons=mom_result['by_reason'],
                )
                all_results.append(result)
                print_result(result)
            else:
                print(f"   ❌ Not enough 5m data for MOMENTUM")
                
    # --- COMBINED SUMMARY ---
    print(f"\n{'='*65}")
    print(f"COMBINED RESULTS")
    print(f"{'='*65}")
    
    for strat in ['HYDRA', 'GRID', 'VWAP', 'MOMENTUM']:
        sr = [r for r in all_results if r.strategy == strat]
        if sr:
            total_pnl = sum(r.total_pnl for r in sr)
            total_trades = sum(r.total_trades for r in sr)
            total_wins = sum(r.winners for r in sr)
            wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
            emoji = {'HYDRA': '🐉', 'GRID': '📐', 'VWAP': '🔮', 'MOMENTUM': '🚀'}[strat]
            print(f"  {emoji} {strat:6s}: ${total_pnl:>10,.2f}  "
                  f"({total_trades} trades, {wr:.0f}% WR)")
    
    grand_total = sum(r.total_pnl for r in all_results)
    print(f"  {'─'*45}")
    print(f"  {'TOTAL':>9s}: ${grand_total:>10,.2f}  "
          f"(Return: {grand_total/capital*100:.1f}%)")
    print(f"  Final equity: ${capital + grand_total:,.2f}")
    
    if grand_total > 0:
        days_tested = max(r.days_tested for r in all_results) if all_results else 1
        daily_avg = grand_total / max(days_tested, 1)
        print(f"\n  📈 Est. daily: ${daily_avg:.2f} | Monthly: ${daily_avg*30:.2f}")
    
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description='Strategy Suite Backtester')
    parser.add_argument('--days', '-d', type=int, default=90,
                        help='Days of history (default: 90)')
    parser.add_argument('--capital', '-c', type=float, default=STARTING_CAPITAL,
                        help=f'Starting capital (default: {STARTING_CAPITAL})')
    parser.add_argument('--strategy', '-s', nargs='+',
                        choices=['hydra', 'grid', 'vwap', 'momentum'],
                        default=['hydra', 'grid', 'momentum'],
                        help='Strategies to test (default: all)')
    parser.add_argument('--symbol', nargs='+', default=DEFAULT_ASSETS,
                        help='Symbols to test')
    parser.add_argument('--debug', action='store_true',
                        help='Show HYDRA debug info')
    
    args = parser.parse_args()
    run_backtest(args.symbol, args.strategy, args.days, args.capital, args.debug)


if __name__ == '__main__':
    main()
