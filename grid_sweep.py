#!/usr/bin/env python3
"""
Grid Adaptive Parameter Sweep
================================
Find the best grid settings using full year of historical data.

Usage:
    python3.10 grid_sweep.py
    python3.10 grid_sweep.py --symbol SOL/USD
"""

import sys
import argparse
from pathlib import Path
from itertools import product
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "historical"
COMMISSION = 0.0015


def backtest_grid_config(df, capital, num_grids, min_range, max_range, 
                         atr_mult, rebal_edge, rebal_interval_days):
    """Backtest one grid configuration."""
    grid_capital = capital * 0.85
    
    if len(df) < 100:
        return {'pnl': 0, 'trades': 0}
    
    # Candles per day
    if len(df) > 1:
        td = (df.index[1] - df.index[0]).total_seconds()
        cpd = int(86400 / td) if td > 0 else 96
    else:
        cpd = 96
    
    rebalance_interval = cpd * rebal_interval_days
    buy_levels = num_grids // 2
    
    trades = []
    total_profit = 0
    equity = capital
    
    grid_center = 0
    grid_spacing = 0
    grid_buys = {}
    order_qty = 0
    current_range_pct = 0.08
    
    for i in range(1, len(df)):
        price = df.iloc[i]['close']
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        
        needs_setup = (i % rebalance_interval == 1 or grid_center == 0)
        
        # Adaptive edge rebalance
        if grid_center > 0 and grid_spacing > 0:
            lower = grid_center * (1 - current_range_pct)
            upper = grid_center * (1 + current_range_pct)
            range_size = upper - lower
            if range_size > 0:
                if price < lower + range_size * rebal_edge or price > upper - range_size * rebal_edge:
                    needs_setup = True
        
        if needs_setup:
            grid_buys = {}
            grid_center = price
            
            # ATR-based range
            if i > 96:
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
                    current_range_pct = max(min_range, min(max_range, atr_pct * atr_mult))
            
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
        
        for lp, has_inv in list(grid_buys.items()):
            if not has_inv and low <= lp:
                grid_buys[lp] = True
        
        for lp, has_inv in list(grid_buys.items()):
            if has_inv:
                sp = lp + grid_spacing
                if high >= sp:
                    profit = grid_spacing * order_qty
                    fee = (lp + sp) * order_qty * COMMISSION
                    net = profit - fee
                    total_profit += net
                    equity += net
                    trades.append(net)
                    grid_buys[lp] = False
    
    days = (df.index[-1] - df.index[0]).days or 1
    
    return {
        'pnl': total_profit,
        'trades': len(trades),
        'daily': total_profit / days,
        'monthly': total_profit / days * 30,
        'annual': total_profit / days * 365,
        'cycles_day': len(trades) / days,
        'avg_cycle': total_profit / len(trades) if trades else 0,
    }


def sweep_symbol(symbol, df, capital=1000):
    """Sweep all parameter combinations for one symbol."""
    
    print(f"\n  {symbol}: {len(df)} candles, {(df.index[-1] - df.index[0]).days} days")
    
    # Parameter grid
    num_grids_list = [8, 10, 12, 15, 20]
    min_ranges = [0.03, 0.05, 0.06, 0.08]
    max_ranges = [0.10, 0.12, 0.15, 0.20]
    atr_mults = [4, 5, 6, 8, 10]
    rebal_edges = [0.10, 0.15, 0.20, 0.25, 0.30]
    rebal_days = [3, 5, 7, 14]
    
    # Filter: max must be > min
    combos = []
    for ng, mn, mx, am, re, rd in product(num_grids_list, min_ranges, max_ranges, 
                                           atr_mults, rebal_edges, rebal_days):
        if mx > mn:
            combos.append((ng, mn, mx, am, re, rd))
    
    total = len(combos)
    print(f"  Testing {total} combinations...")
    
    results = []
    for idx, (ng, mn, mx, am, re, rd) in enumerate(combos):
        if idx % 500 == 0:
            print(f"    {idx}/{total}...", end='\r')
        
        r = backtest_grid_config(df, capital, ng, mn, mx, am, re, rd)
        r.update({
            'symbol': symbol,
            'num_grids': ng,
            'min_range': mn,
            'max_range': mx,
            'atr_mult': am,
            'rebal_edge': re,
            'rebal_days': rd,
        })
        results.append(r)
    
    print(f"    Done ({total} tested)" + " " * 20)
    return results


def main():
    parser = argparse.ArgumentParser(description='Grid Parameter Sweep')
    parser.add_argument('--symbol', '-s', nargs='+', 
                        default=['SOL/USD', 'ETH/USD', 'LINK/USD', 'AVAX/USD'])
    parser.add_argument('--capital', '-c', type=float, default=1000)
    args = parser.parse_args()
    
    print("=" * 75)
    print("ADAPTIVE GRID PARAMETER SWEEP — FULL YEAR")
    print("=" * 75)
    
    all_results = []
    
    for symbol in args.symbol:
        clean = symbol.replace('/', '_')
        csv_path = DATA_DIR / f"{clean}_15m.csv"
        
        if not csv_path.exists():
            print(f"\n  ❌ {csv_path} not found — run download_data.py first")
            continue
        
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        results = sweep_symbol(symbol, df, args.capital)
        all_results.extend(results)
    
    if not all_results:
        print("No results!")
        return
    
    df_r = pd.DataFrame(all_results)
    
    # Best per symbol
    print(f"\n{'='*75}")
    print("BEST CONFIG PER SYMBOL")
    print(f"{'='*75}")
    
    for symbol in args.symbol:
        sym = df_r[(df_r['symbol'] == symbol) & (df_r['trades'] >= 100)]
        if len(sym) == 0:
            print(f"\n  {symbol}: No configs with 100+ trades")
            continue
        
        best = sym.loc[sym['pnl'].idxmax()]
        print(f"\n  🏆 {symbol}:")
        print(f"     num_grids={int(best['num_grids'])}, min_range={best['min_range']:.2f}, "
              f"max_range={best['max_range']:.2f}")
        print(f"     atr_mult={int(best['atr_mult'])}, rebal_edge={best['rebal_edge']:.2f}, "
              f"rebal_days={int(best['rebal_days'])}")
        print(f"     Trades: {int(best['trades'])} | Cycles/day: {best['cycles_day']:.1f}")
        print(f"     P&L: ${best['pnl']:,.2f} | Daily: ${best['daily']:.2f} | "
              f"Monthly: ${best['monthly']:.2f}")
    
    # Top 10 overall
    print(f"\n{'='*75}")
    print("TOP 10 CONFIGS ACROSS ALL SYMBOLS")
    print(f"{'='*75}")
    
    top = df_r[df_r['trades'] >= 100].nlargest(10, 'pnl')
    
    print(f"\n{'Symbol':<10} {'Grids':>5} {'Min%':>5} {'Max%':>5} {'ATR×':>4} {'Edge':>5} "
          f"{'Days':>4} | {'Trades':>6} {'$/day':>7} {'$/mo':>8} {'$/yr':>9}")
    print("-" * 85)
    
    for _, r in top.iterrows():
        print(f"  {r['symbol']:<8} {int(r['num_grids']):>5} {r['min_range']*100:>4.0f}% "
              f"{r['max_range']*100:>4.0f}% {int(r['atr_mult']):>4} {r['rebal_edge']:>5.2f} "
              f"{int(r['rebal_days']):>4} | {int(r['trades']):>5} ${r['daily']:>6.2f} "
              f"${r['monthly']:>7.2f} ${r['annual']:>8.2f}")
    
    # Universal best (sum P&L across all symbols for same params)
    print(f"\n{'='*75}")
    print("BEST UNIVERSAL CONFIG (works across all symbols)")
    print(f"{'='*75}")
    
    param_cols = ['num_grids', 'min_range', 'max_range', 'atr_mult', 'rebal_edge', 'rebal_days']
    grouped = df_r[df_r['trades'] >= 50].groupby(param_cols).agg({
        'pnl': 'sum', 'trades': 'sum', 'daily': 'sum',
    }).reset_index()
    
    top_uni = grouped.nlargest(5, 'pnl')
    
    print(f"\n{'Grids':>5} {'Min%':>5} {'Max%':>5} {'ATR×':>4} {'Edge':>5} {'Days':>4} | "
          f"{'Trades':>6} {'Total $/day':>11} {'Total $/yr':>11}")
    print("-" * 70)
    
    for _, r in top_uni.iterrows():
        annual = r['daily'] * 365
        print(f"  {int(r['num_grids']):>5} {r['min_range']*100:>4.0f}% {r['max_range']*100:>4.0f}% "
              f"{int(r['atr_mult']):>4} {r['rebal_edge']:>5.2f} {int(r['rebal_days']):>4} | "
              f"{int(r['trades']):>5} ${r['daily']:>10.2f} ${annual:>10.2f}")
    
    if len(top_uni) > 0:
        w = top_uni.iloc[0]
        print(f"\n  🏆 RECOMMENDED UNIVERSAL CONFIG:")
        print(f"     num_grids = {int(w['num_grids'])}")
        print(f"     min_range_pct = {w['min_range']}")
        print(f"     max_range_pct = {w['max_range']}")
        print(f"     atr_mult = {int(w['atr_mult'])}  (optimal_range = atr_pct × {int(w['atr_mult'])})")
        print(f"     rebal_edge = {w['rebal_edge']}  (rebalance when price within {w['rebal_edge']*100:.0f}% of edge)")
        print(f"     rebal_days = {int(w['rebal_days'])}  (force rebalance every {int(w['rebal_days'])} days)")
        print(f"     Expected: ${w['daily']:.2f}/day across all pairs on $1k")


if __name__ == '__main__':
    main()
