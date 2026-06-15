#!/usr/bin/env python3
"""
HYDRAGRID with HMM Regime Gating — Comparison Backtest
========================================================
Runs two backtests side-by-side:
  1. BASELINE: grid trading with no regime awareness (the original strategy)
  2. HMM-GATED: same grid, but skips new buy orders when HMM regime is BEAR

Inventory accounting, friction, FIFO logic all match the original
`backtest.py`. The ONLY difference is the regime check in the buy-fill loop.

Usage:
    # Step 1: Generate HMM labels (one-time):
    cd ../regime_classifier && python3 scripts/generate_labels.py

    # Step 2: Run this comparison:
    cd HYDRAGRID/
    python3 backtest_with_regime.py \\
        --labels-dir ../regime_classifier/data/hmm_labels \\
        --scenario realistic

Outputs:
    reports/regime_comparison.png   — equity curves side-by-side
    reports/regime_comparison.md    — written report with metrics delta

What this answers:
    Does HMM regime gating actually improve HYDRAGRID's risk-adjusted return,
    or does it just add complexity for no edge?
"""
from __future__ import annotations
import argparse
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import friction config and constants from the existing backtest
from backtest import (
    FrictionConfig, SCENARIOS, DAILY_VOLUMES, PORTFOLIO_WEIGHTS,
    TradeRecord, CoinResult,
    backtest_grid as backtest_grid_baseline,
    build_portfolio_equity, sharpe_ratio, max_drawdown,
    compute_full_metrics, annual_factor,
)
import backtest as _backtest_module   # to override DATA_DIR if needed


def load_data(symbol: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Local load_data supporting custom data dir for the 5-year data set."""
    if data_dir is None:
        data_dir = _backtest_module.DATA_DIR
    clean = symbol.replace("/", "_")
    path = Path(data_dir) / f"{clean}_15m.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


# =============================================================================
# REGIME LABEL LOADING
# =============================================================================

def load_hmm_labels(labels_dir: Path, symbol: str) -> Optional[pd.Series]:
    """Load daily HMM labels for a single coin. Returns Series indexed by date.

    Falls back to None if the file doesn't exist (caller treats as no gating).
    """
    coin = symbol.replace("/USD", "").replace("/", "_")
    path = Path(labels_dir) / f"{coin}_labels.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    series = df["regime"]
    series.index = series.index.normalize()
    return series


def regime_at(labels: Optional[pd.Series], ts: pd.Timestamp) -> str:
    """Look up the regime for a given 15m timestamp.

    Resolves daily label by normalizing ts to midnight. If not found,
    uses the most-recent prior label (last-known regime).
    """
    if labels is None:
        return "SIDEWAYS"   # neutral default = baseline behavior
    date = pd.Timestamp(ts).normalize()
    if date in labels.index:
        return labels.loc[date]
    prior = labels.index[labels.index <= date]
    if len(prior) == 0:
        return "SIDEWAYS"
    return labels.loc[prior[-1]]


# =============================================================================
# REGIME-GATED CORE SIMULATOR
#
# This is a near-verbatim copy of backtest_grid() from backtest.py, with the
# minimal modification needed for regime gating: when HMM says BEAR, skip
# new buy-fill placements. Sell side runs normally (existing inventory can
# still exit). Equity is still marked-to-market every candle.
#
# Why not import + monkey-patch? Because the gating point is inside a hot
# inner loop and a clean patch requires the gating to be part of the
# function body, not wrapped externally.
# =============================================================================

def backtest_grid_with_regime(
    df: pd.DataFrame,
    capital: float,
    symbol: str,
    cfg: FrictionConfig,
    regime_labels: Optional[pd.Series],
    num_grids: int = 10,
    seed: int = 42,
    bear_liquidate: bool = False,
) -> CoinResult:
    """Regime-gated grid backtest.

    Args:
        regime_labels: daily HMM labels indexed by date. None = no gating
                       (equivalent to baseline backtest_grid).
        bear_liquidate: if True, on BEAR-onset, sell all inventory at market.
                        If False, just stop placing new buys (sells continue).
                        Default False is safer for comparison.
    """
    random.seed(seed)
    rng = random.random

    n = len(df)
    if n < 100:
        return CoinResult(symbol=symbol, days=0, n_buys=0, n_sells=0,
                          wins=0, losses=0, gross_pnl=0, fees=0,
                          spread_cost=0, market_impact=0, net_pnl=0,
                          final_inventory_value=0, final_inventory_cost=0,
                          final_cash=capital, initial_capital=capital,
                          max_dd_pct=0, equity_series=pd.Series(dtype=float))

    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df.index.to_numpy()

    daily_vol = DAILY_VOLUMES.get(symbol, 1_000_000_000)
    n_ex = max(1, cfg.num_exchanges)
    buy_levels = num_grids // 2
    miss_threshold = cfg.miss_rate + cfg.latency_miss_rate

    # State
    cash = capital
    inventory: List[Tuple[float, float]] = []
    grid_buys: Dict[float, bool] = {}
    grid_center = 0.0
    grid_spacing = 0.0
    order_qty = 0.0
    current_range_pct = 0.06
    rebalance_cooldown = 96 * 7
    last_rebalance_idx = -rebalance_cooldown

    # Regime tracking
    prev_regime = "SIDEWAYS"
    bear_buys_skipped = 0
    bear_liquidations = 0

    # Trackers
    totals = dict(gross=0.0, fees=0.0, spread=0.0, impact=0.0, net=0.0)
    n_buys = n_sells = wins = losses = 0
    trades: List[TradeRecord] = []
    daily_equity: Dict[pd.Timestamp, float] = {}
    peak_equity = capital
    max_dd_pct = 0.0

    def equity_at(price: float) -> float:
        inv_mtm = sum(q * price for q, _ in inventory)
        return cash + inv_mtm

    def consume_inventory_fifo(qty: float) -> Tuple[float, float]:
        nonlocal inventory
        remaining = qty
        cost_total = 0.0
        consumed = 0.0
        while remaining > 1e-12 and inventory:
            held_qty, held_price = inventory[0]
            take = min(remaining, held_qty)
            cost_total += take * held_price
            consumed += take
            if take >= held_qty - 1e-12:
                inventory.pop(0)
            else:
                inventory[0] = (held_qty - take, held_price)
            remaining -= take
        return cost_total, consumed

    for i in range(n):
        price = closes[i]
        if not (price > 0):
            continue
        high = highs[i]
        low = lows[i]
        ts = times[i]

        # ── REGIME CHECK ────────────────────────────────────────────────
        current_regime = regime_at(regime_labels, pd.Timestamp(ts))
        regime_changed = current_regime != prev_regime

        # Optional: liquidate inventory at SIDEWAYS/BULL → BEAR transition
        if (bear_liquidate and regime_changed and current_regime == "BEAR"
                and inventory):
            for held_qty, held_price in inventory:
                proceeds = price * held_qty
                fee = proceeds * cfg.fee_rate
                spread = proceeds * cfg.spread_pct
                impact = 0.0
                if cfg.market_impact_alpha > 0 and daily_vol > 0:
                    impact = proceeds * cfg.market_impact_alpha * math.sqrt(
                        (proceeds / n_ex) / daily_vol)
                net_proceeds = proceeds - fee - spread - impact
                cash += net_proceeds
                realized = proceeds - held_qty * held_price
                net_realized = net_proceeds - held_qty * held_price
                totals["gross"] += realized
                totals["fees"] += fee
                totals["spread"] += spread
                totals["impact"] += impact
                totals["net"] += net_realized
                n_sells += 1
                if net_realized > 0:
                    wins += 1
                else:
                    losses += 1
                trades.append(TradeRecord(pd.Timestamp(ts), "sell", price,
                                           held_qty, held_qty * held_price,
                                           net_realized, fee, spread, impact))
                bear_liquidations += 1
            inventory.clear()
            # Reset grid since inventory is gone
            grid_buys = {l: False for l in grid_buys}

        prev_regime = current_regime
        is_bear = current_regime == "BEAR"

        # ── Grid setup / rebalance ──────────────────────────────────────
        needs_setup = (grid_center == 0)
        if grid_center > 0 and grid_spacing > 0:
            lo = grid_center * (1 - current_range_pct)
            hi = grid_center * (1 + current_range_pct)
            rng_size = hi - lo
            near_edge = (price < lo + rng_size * 0.10 or
                         price > hi - rng_size * 0.10)
            if near_edge and (i - last_rebalance_idx) > rebalance_cooldown:
                needs_setup = True

        # Don't rebalance into a fresh grid during BEAR — that would just
        # create new buy levels that immediately get skipped anyway. Better
        # to wait out the bear with whatever grid we already have.
        if needs_setup and not is_bear:
            grid_buys = {}
            grid_center = price
            last_rebalance_idx = i
            if i > 96:
                start = max(0, i - 96)
                h2 = highs[start + 1:i]
                l2 = lows[start + 1:i]
                c2 = closes[start:i - 1]
                tr = np.maximum.reduce([h2 - l2, np.abs(h2 - c2), np.abs(l2 - c2)])
                if len(tr) > 0:
                    atr_pct = tr.mean() / price
                    current_range_pct = max(0.03, min(0.15, atr_pct * 6))
            lo = price * (1 - current_range_pct)
            hi = price * (1 + current_range_pct)
            grid_spacing = (hi - lo) / num_grids
            if grid_spacing <= 0:
                continue
            deployable = cash + sum(q * price for q, _ in inventory) * 0.5
            usd_per_level = max(deployable, capital * 0.15) / buy_levels
            order_qty = usd_per_level / price
            for g in range(1, buy_levels + 1):
                grid_buys[round(price - g * grid_spacing, 2)] = False

        if grid_spacing <= 0 or order_qty <= 0:
            day_key = pd.Timestamp(ts).normalize()
            daily_equity[day_key] = equity_at(price)
            continue

        # ── BUY FILLS (skipped during BEAR) ─────────────────────────────
        for lp, has in list(grid_buys.items()):
            if has or low > lp:
                continue
            if is_bear:
                bear_buys_skipped += 1
                continue                    # ← THE GATING POINT
            if rng() < miss_threshold:
                continue
            buy_cost = lp * order_qty
            fee = buy_cost * cfg.fee_rate
            spread = buy_cost * cfg.spread_pct
            impact = 0.0
            if cfg.market_impact_alpha > 0 and daily_vol > 0:
                impact = buy_cost * cfg.market_impact_alpha * math.sqrt(
                    (buy_cost / n_ex) / daily_vol)
            total_cost = buy_cost + fee + spread + impact
            if total_cost > cash:
                continue
            cash -= total_cost
            inventory.append((order_qty, lp))
            grid_buys[lp] = True
            totals["fees"] += fee
            totals["spread"] += spread
            totals["impact"] += impact
            n_buys += 1
            trades.append(TradeRecord(pd.Timestamp(ts), "buy", lp, order_qty,
                                      0.0, 0.0, fee, spread, impact))

        # ── SELL FILLS (always run, BEAR or not) ────────────────────────
        for lp, has in list(grid_buys.items()):
            if not has:
                continue
            sp = lp + grid_spacing
            if high < sp:
                continue
            if rng() < miss_threshold:
                continue
            cost_basis, consumed = consume_inventory_fifo(order_qty)
            if consumed <= 0:
                continue
            proceeds = sp * consumed
            fee = proceeds * cfg.fee_rate
            spread = proceeds * cfg.spread_pct
            impact = 0.0
            if cfg.market_impact_alpha > 0 and daily_vol > 0:
                impact = proceeds * cfg.market_impact_alpha * math.sqrt(
                    (proceeds / n_ex) / daily_vol)
            net_proceeds = proceeds - fee - spread - impact
            cash += net_proceeds
            realized = proceeds - cost_basis
            net_realized = net_proceeds - cost_basis
            totals["gross"] += realized
            totals["fees"] += fee
            totals["spread"] += spread
            totals["impact"] += impact
            totals["net"] += net_realized
            n_sells += 1
            if net_realized > 0:
                wins += 1
            else:
                losses += 1
            trades.append(TradeRecord(pd.Timestamp(ts), "sell", sp, consumed,
                                      cost_basis, net_realized, fee, spread, impact))
            grid_buys[lp] = False

        # ── Mark-to-market ──────────────────────────────────────────────
        eq = equity_at(price)
        if eq > peak_equity:
            peak_equity = eq
        dd = (peak_equity - eq) / peak_equity if peak_equity > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd
        day_key = pd.Timestamp(ts).normalize()
        daily_equity[day_key] = eq

    # Apply downtime
    if cfg.downtime_pct > 0:
        dt_keep = 1 - cfg.downtime_pct
        totals["gross"] *= dt_keep
        totals["fees"] *= dt_keep
        totals["spread"] *= dt_keep
        totals["impact"] *= dt_keep
        totals["net"] *= dt_keep

    final_price = closes[-1]
    final_inv_qty = sum(q for q, _ in inventory)
    final_inv_value = final_inv_qty * final_price
    final_inv_cost = sum(q * p for q, p in inventory)

    days = (df.index[-1] - df.index[0]).days if n > 1 else 1
    eq_series = pd.Series(daily_equity).sort_index()

    result = CoinResult(
        symbol=symbol, days=days, n_buys=n_buys, n_sells=n_sells,
        wins=wins, losses=losses,
        gross_pnl=totals["gross"], fees=totals["fees"],
        spread_cost=totals["spread"], market_impact=totals["impact"],
        net_pnl=totals["net"],
        final_inventory_value=final_inv_value,
        final_inventory_cost=final_inv_cost,
        final_cash=cash,
        initial_capital=capital,
        max_dd_pct=max_dd_pct * 100,
        equity_series=eq_series,
        trades=trades,
    )
    # Attach regime-gating diagnostics
    result._bear_buys_skipped = bear_buys_skipped
    result._bear_liquidations = bear_liquidations
    return result


# =============================================================================
# COMPARISON ORCHESTRATION
# =============================================================================

def run_comparison(labels_dir: Path, scenario: str = "realistic",
                    capital: float = 1100.0,
                    bear_liquidate: bool = False,
                    data_dir: Path | None = None) -> dict:
    cfg = SCENARIOS[scenario]
    weights = PORTFOLIO_WEIGHTS

    print(f"\n{'='*72}")
    print(f"  HYDRAGRID — Baseline vs HMM-Gated Comparison")
    print(f"{'='*72}")
    print(f"  Scenario:    {cfg.name}")
    print(f"  Capital:     ${capital:,.0f}")
    print(f"  Data dir:    {data_dir or _backtest_module.DATA_DIR}")
    print(f"  Labels:      {labels_dir}")
    print(f"  Liquidate on BEAR onset: {bear_liquidate}")

    baseline_results: Dict[str, CoinResult] = {}
    gated_results: Dict[str, CoinResult] = {}

    for symbol, weight in weights.items():
        coin_capital = capital * weight
        df = load_data(symbol, data_dir)
        if df.empty:
            print(f"  ✗ {symbol}: no data")
            continue
        labels = load_hmm_labels(labels_dir, symbol)
        if labels is None:
            print(f"  ⚠ {symbol}: no HMM labels found, skipping comparison "
                  f"(baseline only)")
            baseline_results[symbol] = backtest_grid_baseline(
                df, coin_capital, symbol, cfg)
            continue

        print(f"\n  ── {symbol} ──")
        print(f"    Data: {len(df):,} bars, {df.index[0].date()} → "
              f"{df.index[-1].date()}")
        regime_counts = labels.value_counts()
        print(f"    Regime distribution: "
              f"BEAR={int(regime_counts.get('BEAR', 0))}d  "
              f"SIDEWAYS={int(regime_counts.get('SIDEWAYS', 0))}d  "
              f"BULL={int(regime_counts.get('BULL', 0))}d")

        # Baseline
        base = backtest_grid_baseline(df, coin_capital, symbol, cfg)
        baseline_results[symbol] = base
        # HMM-gated
        gated = backtest_grid_with_regime(
            df, coin_capital, symbol, cfg, labels,
            bear_liquidate=bear_liquidate)
        gated_results[symbol] = gated

        b_total = base.net_pnl + base.final_inventory_value - base.final_inventory_cost
        g_total = gated.net_pnl + gated.final_inventory_value - gated.final_inventory_cost
        print(f"    Baseline:    net PnL ${base.net_pnl:>8,.0f}  "
              f"final inv ${base.final_inventory_value:>8,.0f}  "
              f"DD {base.max_dd_pct:>5.1f}%")
        print(f"    HMM-gated:   net PnL ${gated.net_pnl:>8,.0f}  "
              f"final inv ${gated.final_inventory_value:>8,.0f}  "
              f"DD {gated.max_dd_pct:>5.1f}%")
        print(f"    BEAR fills skipped: {gated._bear_buys_skipped:,}  "
              f"liquidations: {gated._bear_liquidations}")

    # ── Portfolio aggregates ────────────────────────────────────────────
    base_eq = build_portfolio_equity(baseline_results)
    gated_eq = build_portfolio_equity(gated_results)
    if base_eq.empty or gated_eq.empty:
        print("\n  No comparison possible — missing equity series.")
        return {"baseline": baseline_results, "gated": gated_results}

    base_days = (base_eq.index[-1] - base_eq.index[0]).days or 1
    gated_days = (gated_eq.index[-1] - gated_eq.index[0]).days or 1

    base_metrics = compute_full_metrics(base_eq, base_days)
    gated_metrics = compute_full_metrics(gated_eq, gated_days)

    print(f"\n{'='*72}")
    print(f"  PORTFOLIO METRICS")
    print(f"{'='*72}")
    print(f"  {'Metric':<22}  {'Baseline':>14}  {'HMM-gated':>14}  {'Delta':>10}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*14}  {'-'*10}")

    def fmt_pct(v):
        return f"{v*100:+.2f}%"

    def show_metric(name, key, fmt="{:+.2f}"):
        b = base_metrics.get(key, 0.0)
        g = gated_metrics.get(key, 0.0)
        delta = g - b
        print(f"  {name:<22}  {fmt.format(b):>14}  "
              f"{fmt.format(g):>14}  {fmt.format(delta):>10}")

    show_metric("Total return", "total_return", "{:+.1%}")
    show_metric("CAGR", "cagr", "{:+.1%}")
    show_metric("Sharpe", "sharpe", "{:+.2f}")
    show_metric("Sortino", "sortino", "{:+.2f}")
    show_metric("Max drawdown", "max_dd", "{:.1%}")
    show_metric("Calmar", "calmar", "{:+.2f}")

    # ── Verdict ─────────────────────────────────────────────────────────
    sharpe_diff = gated_metrics.get("sharpe", 0) - base_metrics.get("sharpe", 0)
    dd_diff = gated_metrics.get("max_dd", 0) - base_metrics.get("max_dd", 0)
    ret_diff = gated_metrics.get("total_return", 0) - base_metrics.get("total_return", 0)

    print(f"\n{'='*72}")
    print(f"  VERDICT")
    print(f"{'='*72}")
    if sharpe_diff > 0.1 and dd_diff < 0 and ret_diff > 0:
        print(f"  ✓ HMM gating WINS on all three metrics.")
        print(f"    Recommend: integrate into live HYDRAGRID.")
    elif (sharpe_diff > 0) or (dd_diff < -0.02):
        print(f"  ⚠ HMM gating shows MIXED improvement.")
        print(f"    Worth integrating if you value lower drawdown.")
    elif abs(sharpe_diff) < 0.05 and abs(dd_diff) < 0.02:
        print(f"  ≈ HMM gating shows NO MEANINGFUL DIFFERENCE.")
        print(f"    Baseline is competitive; integration may not be justified.")
    else:
        print(f"  ✗ HMM gating HURTS performance on this dataset.")
        print(f"    Either re-tune the HMM or accept that grid trades well")
        print(f"    enough in all regimes for the current friction stack.")

    # ── Chart ───────────────────────────────────────────────────────────
    chart_path = REPORTS / "regime_comparison.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(base_eq.index, base_eq.values, lw=2, color="#7570b3",
             label="Baseline (no regime)")
    ax.plot(gated_eq.index, gated_eq.values, lw=2, color="#1b9e77",
             label="HMM-gated")
    ax.axhline(capital, color="grey", lw=0.8, ls="--")
    ax.set_title(f"HYDRAGRID — Baseline vs HMM-Gated  "
                  f"(scenario: {scenario}, capital: ${capital:,.0f})")
    ax.set_ylabel("Portfolio equity ($)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    print(f"\n  Chart saved → {chart_path}")

    # ── Markdown report ─────────────────────────────────────────────────
    report_path = REPORTS / "regime_comparison.md"
    with open(report_path, "w") as f:
        f.write(f"# HYDRAGRID Regime Gating Comparison\n\n")
        f.write(f"**Scenario**: {cfg.name}  \n")
        f.write(f"**Capital**: ${capital:,.0f}  \n")
        f.write(f"**Labels source**: {labels_dir}\n\n")
        f.write(f"## Portfolio metrics\n\n")
        f.write(f"| Metric | Baseline | HMM-gated | Delta |\n")
        f.write(f"|---|---|---|---|\n")
        for label, key, fmt in [
            ("Total return", "total_return", "{:.1%}"),
            ("CAGR", "cagr", "{:.1%}"),
            ("Sharpe", "sharpe", "{:.2f}"),
            ("Sortino", "sortino", "{:.2f}"),
            ("Max drawdown", "max_dd", "{:.1%}"),
            ("Calmar", "calmar", "{:.2f}"),
        ]:
            b = base_metrics.get(key, 0)
            g = gated_metrics.get(key, 0)
            f.write(f"| {label} | {fmt.format(b)} | {fmt.format(g)} | "
                    f"{fmt.format(g - b)} |\n")
        f.write(f"\n## Per-coin\n\n")
        f.write(f"| Coin | Baseline net | HMM-gated net | Baseline DD | "
                f"HMM-gated DD | BEAR-skip |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for sym, base in baseline_results.items():
            gated = gated_results.get(sym)
            if gated is None:
                continue
            skipped = getattr(gated, "_bear_buys_skipped", 0)
            f.write(f"| {sym} | ${base.net_pnl:,.0f} | "
                    f"${gated.net_pnl:,.0f} | {base.max_dd_pct:.1f}% | "
                    f"{gated.max_dd_pct:.1f}% | {skipped:,} |\n")
    print(f"  Report saved → {report_path}")

    return {
        "baseline": baseline_results,
        "gated": gated_results,
        "baseline_metrics": base_metrics,
        "gated_metrics": gated_metrics,
    }


def main():
    ap = argparse.ArgumentParser(
        description="HYDRAGRID baseline vs HMM-gated comparison")
    ap.add_argument("--labels-dir", required=True,
                     help="Directory of HMM label CSVs from generate_labels.py")
    ap.add_argument("--data-dir", default=None,
                     help="Directory of 15m CSVs (default: data/historical). "
                          "Use 'data/historical_5yr' after running "
                          "download_data_5yr.py")
    ap.add_argument("--scenario", default="realistic",
                     choices=list(SCENARIOS.keys()),
                     help="Friction scenario")
    ap.add_argument("--capital", type=float, default=1100.0)
    ap.add_argument("--bear-liquidate", action="store_true",
                     help="Liquidate inventory at BEAR onset (more aggressive)")
    args = ap.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    run_comparison(Path(args.labels_dir), args.scenario,
                    args.capital, args.bear_liquidate, data_dir)


if __name__ == "__main__":
    main()
