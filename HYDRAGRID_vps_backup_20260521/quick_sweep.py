#!/usr/bin/env python3
"""Quick targeted grid sweep — ~500 combos, finishes in 3 minutes"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from grid_sweep import backtest_grid_config

DATA_DIR = Path(__file__).parent / "data" / "historical"

results = []
for sym in ['SOL/USD', 'ETH/USD', 'LINK/USD', 'AVAX/USD']:
    clean = sym.replace('/', '_')
    csv = DATA_DIR / f'{clean}_15m.csv'
    if not csv.exists():
        print(f'  ❌ {sym} — no data')
        continue
    df = pd.read_csv(csv, index_col='timestamp', parse_dates=True)
    print(f'{sym}: {len(df)} candles, {(df.index[-1] - df.index[0]).days} days')
    
    count = 0
    for ng in [10, 12, 15]:
        for mn in [0.05, 0.06, 0.08]:
            for mx in [0.12, 0.15, 0.20]:
                for am in [4, 6, 8]:
                    for re in [0.20, 0.25, 0.30]:
                        for rd in [3, 7, 14]:
                            if mx <= mn:
                                continue
                            r = backtest_grid_config(df, 1000, ng, mn, mx, am, re, rd)
                            r['symbol'] = sym
                            r['ng'] = ng
                            r['mn'] = mn
                            r['mx'] = mx
                            r['am'] = am
                            r['re'] = re
                            r['rd'] = rd
                            results.append(r)
                            count += 1
    print(f'  Tested {count} combos')

results.sort(key=lambda x: x['pnl'], reverse=True)

print(f'\n{"="*75}')
print('TOP 10 OVERALL')
print(f'{"="*75}')
print(f'{"Symbol":<10} {"Grids":>5} {"Min%":>5} {"Max%":>5} {"ATRx":>4} {"Edge":>5} {"Days":>4} | {"Trades":>6} {"$/day":>7} {"$/yr":>9}')
print('-' * 75)
for r in results[:10]:
    annual = r['daily'] * 365
    print(f'  {r["symbol"]:<8} {r["ng"]:>5} {r["mn"]*100:>4.0f}% {r["mx"]*100:>4.0f}% {r["am"]:>4} {r["re"]:>5.2f} {r["rd"]:>4} | {r["trades"]:>5} ${r["daily"]:>6.2f} ${annual:>8.2f}')

# Best per symbol
print(f'\n{"="*75}')
print('BEST PER SYMBOL')
print(f'{"="*75}')

for sym in ['SOL/USD', 'ETH/USD', 'LINK/USD', 'AVAX/USD']:
    sym_results = [r for r in results if r['symbol'] == sym and r['trades'] >= 50]
    if not sym_results:
        print(f'\n  {sym}: No results')
        continue
    best = sym_results[0]
    annual = best['daily'] * 365
    print(f'\n  🏆 {sym}:')
    print(f'     num_grids={best["ng"]}, min_range={best["mn"]}, max_range={best["mx"]}')
    print(f'     atr_mult={best["am"]}, rebal_edge={best["re"]}, rebal_days={best["rd"]}')
    print(f'     Trades: {best["trades"]} | ${best["daily"]:.2f}/day | ${annual:.0f}/year')

# Best universal
print(f'\n{"="*75}')
print('BEST UNIVERSAL (same params, summed across all symbols)')
print(f'{"="*75}')

from collections import defaultdict
param_totals = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'daily': 0, 'count': 0})
for r in results:
    key = (r['ng'], r['mn'], r['mx'], r['am'], r['re'], r['rd'])
    param_totals[key]['pnl'] += r['pnl']
    param_totals[key]['trades'] += r['trades']
    param_totals[key]['daily'] += r['daily']
    param_totals[key]['count'] += 1

universal = sorted(param_totals.items(), key=lambda x: x[1]['pnl'], reverse=True)

for (ng, mn, mx, am, re, rd), stats in universal[:5]:
    annual = stats['daily'] * 365
    print(f'  g={ng} mn={mn} mx={mx} am={am} re={re} rd={rd}')
    print(f'    Total: ${stats["pnl"]:.2f} | {stats["trades"]} trades | ${stats["daily"]:.2f}/day | ${annual:.0f}/year')
    print()
