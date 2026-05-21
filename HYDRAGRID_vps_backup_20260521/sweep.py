#!/usr/bin/env python3
"""
Parameter Sweep — Find profitable Momentum & HYDRA settings
Run locally where your CSV data lives.

Usage:
    python3.10 sweep.py
"""

import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.momentum_strategy import backtest_momentum, MomentumConfig

DATA_DIR = Path(__file__).parent / "data" / "historical"


def sweep_momentum():
    """Sweep momentum parameters on full year data."""
    
    print("=" * 80)
    print("MOMENTUM PARAMETER SWEEP — FULL YEAR")
    print("=" * 80)
    
    results = []
    
    for symbol in ['SOL/USD', 'ETH/USD', 'BTC/USD']:
        clean = symbol.replace('/', '_')
        csv_path = DATA_DIR / f"{clean}_5m.csv"
        
        if not csv_path.exists():
            print(f"  ❌ {csv_path} not found")
            continue
        
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        print(f"\n  {symbol}: {len(df)} candles, {(df.index[-1] - df.index[0]).days} days")
        
        # Parameter grid
        triggers = [0.015, 0.02, 0.025, 0.03]       # 1.5% to 3%
        trails = [0.008, 0.01, 0.012, 0.015, 0.02]   # 0.8% to 2%
        volumes = [1.0, 1.3, 1.5, 2.0]                # Volume threshold
        cooldowns = [6, 12, 24]                         # 30min, 1hr, 2hr
        
        total_combos = len(triggers) * len(trails) * len(volumes) * len(cooldowns)
        print(f"  Testing {total_combos} parameter combinations...")
        
        count = 0
        for trig, trail, vol, cool in product(triggers, trails, volumes, cooldowns):
            count += 1
            if count % 50 == 0:
                print(f"    {count}/{total_combos}...", end='\r')
            
            cfg = MomentumConfig(
                trigger_pct=trig,
                trail_pct=trail,
                volume_threshold=vol,
                cooldown_bars=cool,
                max_hold_bars=72,
                lookback_bars=12,
                trade_size_pct=0.08,
                max_trades_per_day=5,
            )
            
            r = backtest_momentum(df, 1000, cfg)
            
            results.append({
                'symbol': symbol,
                'trigger': trig,
                'trail': trail,
                'volume': vol,
                'cooldown': cool,
                'trades': r['total_trades'],
                'wr': r['win_rate'],
                'pnl': r['total_pnl'],
                'dd': r['max_drawdown'],
                'avg_win': r['avg_win'],
                'avg_loss': r['avg_loss'],
            })
        
        print(f"    Done ({count} tested)" + " " * 20)
    
    # Show top results per symbol
    print(f"\n{'='*80}")
    print(f"TOP 10 PROFITABLE CONFIGS (sorted by P&L)")
    print(f"{'='*80}")
    
    df_results = pd.DataFrame(results)
    
    # Filter: need at least 20 trades and positive P&L
    profitable = df_results[(df_results['pnl'] > 0) & (df_results['trades'] >= 20)]
    
    if len(profitable) == 0:
        print("\n  ❌ No profitable configurations found with 20+ trades")
        print("\n  Showing least-bad configs:")
        profitable = df_results[df_results['trades'] >= 20].nlargest(10, 'pnl')
    else:
        profitable = profitable.nlargest(10, 'pnl')
    
    print(f"\n{'Symbol':<10} {'Trig':>5} {'Trail':>6} {'Vol':>4} {'Cool':>5} | "
          f"{'Trades':>6} {'WR%':>5} {'P&L':>9} {'DD%':>5}")
    print("-" * 80)
    
    for _, r in profitable.iterrows():
        emoji = "✅" if r['pnl'] > 0 else "❌"
        print(f"{emoji} {r['symbol']:<8} {r['trigger']:>5.3f} {r['trail']:>6.3f} "
              f"{r['volume']:>4.1f} {r['cooldown']:>5.0f} | "
              f"{r['trades']:>5.0f}  {r['wr']:>4.0f}%  ${r['pnl']:>8.2f} {r['dd']:>5.1f}%")
    
    # Best config per symbol
    print(f"\n{'='*80}")
    print(f"BEST CONFIG PER SYMBOL")
    print(f"{'='*80}")
    
    for symbol in ['SOL/USD', 'ETH/USD', 'BTC/USD']:
        sym_data = df_results[
            (df_results['symbol'] == symbol) & 
            (df_results['trades'] >= 20)
        ]
        if len(sym_data) == 0:
            continue
        
        best = sym_data.loc[sym_data['pnl'].idxmax()]
        emoji = "✅" if best['pnl'] > 0 else "❌"
        print(f"\n  {emoji} {symbol}:")
        print(f"     trigger={best['trigger']:.3f}, trail={best['trail']:.3f}, "
              f"volume={best['volume']:.1f}, cooldown={best['cooldown']:.0f}")
        print(f"     Trades: {best['trades']:.0f} | WR: {best['wr']:.0f}% | "
              f"P&L: ${best['pnl']:.2f} | DD: {best['dd']:.1f}%")
    
    # Cross-symbol: find params that work on ALL symbols
    print(f"\n{'='*80}")
    print(f"UNIVERSAL CONFIG (works across all symbols)")
    print(f"{'='*80}")
    
    # Group by params, sum P&L across symbols
    param_cols = ['trigger', 'trail', 'volume', 'cooldown']
    grouped = df_results.groupby(param_cols).agg({
        'pnl': 'sum',
        'trades': 'sum',
        'wr': 'mean',
        'dd': 'max',
    }).reset_index()
    
    grouped = grouped[grouped['trades'] >= 50]
    best_universal = grouped.nlargest(5, 'pnl')
    
    print(f"\n{'Trig':>5} {'Trail':>6} {'Vol':>4} {'Cool':>5} | "
          f"{'Trades':>6} {'WR%':>5} {'Total P&L':>10} {'Max DD':>7}")
    print("-" * 65)
    
    for _, r in best_universal.iterrows():
        emoji = "✅" if r['pnl'] > 0 else "❌"
        print(f"{emoji} {r['trigger']:>5.3f} {r['trail']:>6.3f} "
              f"{r['volume']:>4.1f} {r['cooldown']:>5.0f} | "
              f"{r['trades']:>5.0f}  {r['wr']:>4.0f}%  ${r['pnl']:>9.2f} {r['dd']:>6.1f}%")
    
    if len(best_universal) > 0:
        winner = best_universal.iloc[0]
        print(f"\n  🏆 RECOMMENDED CONFIG:")
        print(f"     trigger_pct={winner['trigger']:.3f}")
        print(f"     trail_pct={winner['trail']:.3f}")
        print(f"     volume_threshold={winner['volume']:.1f}")
        print(f"     cooldown_bars={int(winner['cooldown'])}")


def sweep_hydra():
    """Sweep HYDRA parameters on full year 15m data. Slower — fewer combos."""
    from src.hydra_spot import HydraSpotStrategy, HydraConfig
    
    print("\n" + "=" * 80)
    print("HYDRA PARAMETER SWEEP — FULL YEAR")
    print("(This takes ~10-20 min — fewer combos because each run is slow)")
    print("=" * 80)
    
    results = []
    
    for symbol in ['SOL/USD', 'ETH/USD', 'BTC/USD']:
        clean = symbol.replace('/', '_')
        csv_path = DATA_DIR / f"{clean}_15m.csv"
        
        if not csv_path.exists():
            print(f"  ❌ {csv_path} not found")
            continue
        
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        print(f"\n  {symbol}: {len(df)} candles, {(df.index[-1] - df.index[0]).days} days")
        
        # Key parameters to sweep (keep it small — each run is slow)
        quality_gates = [0.85, 0.90, 0.95]
        regime_confs = [0.65, 0.75, 0.85]
        ttl_multipliers = [0.5, 1.0, 1.5]  # Relative to current TTLs
        risk_pcts = [0.05, 0.10]
        
        total = len(quality_gates) * len(regime_confs) * len(ttl_multipliers) * len(risk_pcts)
        print(f"  Testing {total} configs...")
        
        count = 0
        for qual, regime, ttl_mult, risk in product(quality_gates, regime_confs, ttl_multipliers, risk_pcts):
            count += 1
            print(f"    {count}/{total} (q={qual}, r={regime}, ttl={ttl_mult}x, risk={risk})...", end='\r')
            
            cfg = HydraConfig()
            cfg.min_quality_score = qual
            cfg.min_regime_confidence = regime
            cfg.snapback_ttl = int(36 * ttl_mult)
            cfg.compress_ttl = int(48 * ttl_mult)
            cfg.pullback_ttl = int(40 * ttl_mult)
            cfg.momentum_ttl = int(48 * ttl_mult)
            cfg.base_risk_pct = risk
            
            strategy = HydraSpotStrategy(cfg)
            strategy.equity = 1000
            strategy.starting_equity = 1000
            strategy.peak_equity = 1000
            strategy.day_start_equity = 1000
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            warmup = 150
            for i in range(warmup, len(df)):
                candles = df.iloc[:i+1].copy()
                strategy.on_bar(symbol, candles, i)
            
            # Close open positions
            if strategy.positions:
                final_price = float(df.iloc[-1]['close'])
                for pos in strategy.positions:
                    if pos.active:
                        strategy._close_position(pos, final_price, 'END')
            
            trade_results = strategy.trade_results
            total_trades = len(trade_results)
            wins = sum(trade_results) if trade_results else 0
            pnl = strategy.equity - 1000
            
            results.append({
                'symbol': symbol,
                'quality': qual,
                'regime': regime,
                'ttl_mult': ttl_mult,
                'risk': risk,
                'trades': total_trades,
                'wins': wins,
                'wr': (wins / total_trades * 100) if total_trades > 0 else 0,
                'pnl': pnl,
            })
        
        print(f"    Done ({count} tested)" + " " * 40)
    
    # Results
    df_r = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"HYDRA — BEST CONFIGS PER SYMBOL")
    print(f"{'='*80}")
    
    for symbol in ['SOL/USD', 'ETH/USD', 'BTC/USD']:
        sym = df_r[(df_r['symbol'] == symbol) & (df_r['trades'] >= 3)]
        if len(sym) == 0:
            print(f"\n  {symbol}: No configs with 3+ trades")
            continue
        best = sym.loc[sym['pnl'].idxmax()]
        emoji = "✅" if best['pnl'] > 0 else "❌"
        print(f"\n  {emoji} {symbol}:")
        print(f"     quality={best['quality']}, regime={best['regime']}, "
              f"ttl={best['ttl_mult']}x, risk={best['risk']}")
        print(f"     Trades: {best['trades']:.0f} | Wins: {best['wins']:.0f} | "
              f"WR: {best['wr']:.0f}% | P&L: ${best['pnl']:.2f}")
    
    # Universal
    print(f"\n{'='*80}")
    print(f"HYDRA — UNIVERSAL CONFIG")
    print(f"{'='*80}")
    
    param_cols = ['quality', 'regime', 'ttl_mult', 'risk']
    grouped = df_r.groupby(param_cols).agg({
        'pnl': 'sum', 'trades': 'sum', 'wins': 'sum',
    }).reset_index()
    grouped['wr'] = (grouped['wins'] / grouped['trades'] * 100).fillna(0)
    
    top5 = grouped.nlargest(5, 'pnl')
    
    print(f"\n{'Qual':>5} {'Regime':>7} {'TTL':>5} {'Risk':>5} | "
          f"{'Trades':>6} {'WR%':>5} {'Total P&L':>10}")
    print("-" * 60)
    
    for _, r in top5.iterrows():
        emoji = "✅" if r['pnl'] > 0 else "❌"
        print(f"{emoji} {r['quality']:>5.2f} {r['regime']:>7.2f} {r['ttl_mult']:>5.1f}x "
              f"{r['risk']:>5.2f} | {r['trades']:>5.0f}  {r['wr']:>4.0f}%  ${r['pnl']:>9.2f}")
    
    if len(top5) > 0:
        w = top5.iloc[0]
        if w['pnl'] > 0:
            print(f"\n  🏆 RECOMMENDED HYDRA CONFIG:")
            print(f"     min_quality_score={w['quality']}")
            print(f"     min_regime_confidence={w['regime']}")
            print(f"     TTL multiplier={w['ttl_mult']}x")
            print(f"     base_risk_pct={w['risk']}")
        else:
            print(f"\n  ❌ No profitable HYDRA config found. Kill it.")


if __name__ == '__main__':
    print("Starting Momentum sweep...")
    sweep_momentum()
    
    print("\n\nStarting HYDRA sweep...")
    sweep_hydra()
