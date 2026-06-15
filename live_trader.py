#!/usr/bin/env python3
"""
Live Trader - Multi-Strategy Trading Bot
=========================================
Real-time trading bot that monitors multiple assets with multiple strategies.

Strategies:
    4. GRID: Adaptive grid trading strategy for oscillating assets (SOL, ETH, LINK, AVAX)
    
Usage:
    python live_trader.py                              # Paper trading, all strategies
    python live_trader.py --live                       # Live trading
    python live_trader.py --strategies grid           # Grid only
    python live_trader.py --assets LINK/USD SOL/USD    # Specific assets
    python live_trader.py --interval 60                # Check every 60 seconds

Features:
    - Multi-asset monitoring with per-asset tuned parameters
    - Paper trading mode for testing
    - Live trading via Kraken API
    - Telegram/Discord alerts
    - Position tracking and P&L reporting
    - State persistence across restarts
    - Safety checks and risk management
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
    RISK_PER_TRADE, TIMEFRAME, VERBOSE
)
from src.data_fetcher import DataFetcher
from src.alerts import AlertManager
from src.risk_manager import RiskManager
from src.grid_strategy import GridStrategy, GridConfig

# --- HMM regime classifier ---
from config.settings import REGIME_CLASSIFIER_DIR, REGIME_MODELS_DIR
sys.path.insert(0, str(REGIME_CLASSIFIER_DIR))
from classifier.hmm_classifier import HMMRegimeClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import asset-specific tuned parameters
from config.settings import ASSET_CONFIGS
ASSET_PARAMS = ASSET_CONFIGS

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
# LIVE TRADER
# =============================================================================

class LiveTrader:
    """Grid trading bot with adaptive range and multi-coin support."""
    
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
        
                
        # Connect to exchange
        # For Alpaca, always connect — paper trading uses the paper API endpoint
        if not paper_mode or exchange_id == 'alpaca':
            self.exchange = self._connect_exchange(api_key, api_secret)
        else:
            self.exchange = None
        
        # State tracking        
        self.running = False
        self._daily_summary_date: str = ''
        
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
        
        # --- HMM regime gating ---
        self.classifier = HMMRegimeClassifier(models_dir=str(REGIME_MODELS_DIR))
        self.coin_regime: Dict[str, str] = {s: "SIDEWAYS" for s in self.grids}
        self._regime_date: str = ''   # last UTC date we classified
    
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
    
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    # def fetch_data_15m(self, symbol: str, lookback: int = 200) -> pd.DataFrame:
    #     """Fetch 15m candles for HYDRA strategy."""
    #     try:
    #         df = self.fetcher.fetch_live(
    #             symbol=symbol,
    #             timeframe=HYDRA_TIMEFRAME,
    #             num_candles=lookback
    #         )
            
    #         if len(df) > 0:
    #             last_timestamp = df.index[-1] if hasattr(df.index[-1], 'timestamp') else df.index[-1]
    #             prev_timestamp = self._last_15m_timestamp.get(symbol)
                
    #             if prev_timestamp is None or last_timestamp > prev_timestamp:
    #                 self.hydra_bar_index[symbol] = self.hydra_bar_index.get(symbol, 0) + 1
    #                 self._last_15m_timestamp[symbol] = last_timestamp
                
    #             self.data_cache_15m[symbol] = df
            
    #         return df
            
    #     except Exception as e:
    #         self.alerts.error_alert(f"Error fetching 15m {symbol}: {e}")
    #         return self.data_cache_15m.get(symbol, pd.DataFrame())
    
    # def fetch_data_5m(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
    #     """Fetch 5-minute candles for VWAP strategy."""
    #     try:
    #         df = self.fetcher.fetch_live(symbol=symbol, timeframe='5m', num_candles=lookback)
    #         if len(df) > 0:
    #             self.data_cache_5m[symbol] = df
    #         return df
    #     except Exception as e:
    #         self.alerts.error_alert(f"Error fetching 5m {symbol}: {e}")
    #         return self.data_cache_5m.get(symbol, pd.DataFrame())
    
    def update_regimes(self):
        """Once per UTC day: classify each coin's regime from daily bars."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        if today == self._regime_date:
            return
        for symbol in self.grids:
            coin = symbol.split('/')[0]          # "SOL/USD" -> "SOL"
            try:
                df = self.fetcher.fetch_live(symbol, timeframe='1d', num_candles=250)
                df.columns = [c.lower() for c in df.columns]
                regime = self.classifier.classify_current(coin, df)
                prev = self.coin_regime.get(symbol, "SIDEWAYS")
                self.coin_regime[symbol] = regime
                if regime != prev:
                    self.alerts.status_alert(
                        f"🔀 {symbol} regime {prev} → {regime}")
            except Exception as e:
                self.alerts.error_alert(f"Regime check failed for {symbol}: {e}")
        self._regime_date = today
    
    def _liquidate_grid(self, grid, symbol: str, price: float):
        """BEAR onset: cancel orders AND market-sell held inventory to cash."""
        # 1. cancel all pending orders + deactivate
        grid.shutdown()
        # 2. market-sell every filled level's inventory
        sold_qty = 0.0
        for level in grid.levels:
            if level.status == 'filled':
                try:
                    self.exchange.create_market_order(
                        symbol=symbol, side='sell', amount=grid.order_size_qty)
                    sold_qty += grid.order_size_qty
                    level.status = 'empty'
                    level.order_id = None
                except Exception as e:
                    self.alerts.error_alert(f"Liquidation sell failed {symbol}: {e}")
        if sold_qty > 0:
            self.alerts.status_alert(
                f"🐻 {symbol} BEAR — liquidated {sold_qty:.4f} to cash @ ${price:,.2f}")
        grid._save_state()
            
    # =========================================================================
    # MAIN TRADING LOOP
    # =========================================================================
    
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

                # --- REGIME GATE ---
                if self.coin_regime.get(symbol) == "BEAR":
                    if grid.active:
                        self._liquidate_grid(grid, symbol, current_price)
                    continue          # no grid trading during BEAR regime
                
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
    
    # =========================================================================
    # DAILY P&L SUMMARY
    # =========================================================================
    
    def _check_daily_summary(self):
        """Send daily P&L summary once per day at midnight UTC."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        if today == self._daily_summary_date:
            return
        
        hour = datetime.utcnow().hour
        if hour != 0:
            return
        
        self._daily_summary_date = today
        self._send_daily_summary()
    
    def _send_daily_summary(self):
        """Build and send the daily P&L summary."""
        total_profit = 0
        total_cycles = 0
        coin_lines = []
        
        for symbol, grid in self.grids.items():
            s = grid.get_status()
            profit = s['profit']
            cycles = s['completed']
            total_profit += profit
            total_cycles += cycles
            emoji = "✅" if profit > 0 else "❌" if profit < 0 else "⚪"
            coin_lines.append(
                f"  {emoji} {symbol}: ${profit:,.2f} ({cycles} cycles)"
            )
        
        roi = (total_profit / self.starting_capital * 100) if self.starting_capital > 0 else 0
        coins_text = "\n".join(coin_lines)
        
        message = (
            f"📊 DAILY P&L SUMMARY\n\n"
            f"💰 Total Profit: ${total_profit:,.2f} ({roi:+.1f}%)\n"
            f"🔄 Total Cycles: {total_cycles}\n"
            f"💵 Starting Capital: ${self.starting_capital:,.0f}\n\n"
            f"Per Coin:\n{coins_text}\n\n"
            f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        
        self.alerts.send(message, level="TRADE")
    
    def print_status(self):
        """Print current status."""
        print("\n" + "=" * 70)
        print(f"📊 MULTI-STRATEGY TRADER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        mode = "🟡 PAPER" if self.paper_mode else "🟢 LIVE"
        print(f"\n   Mode: {mode}")
        print(f"   Capital: ${self.capital:,.2f}")
        print(f"   Assets: {', '.join(self.assets)}")
        print(f"   Strategies: {', '.join(s.upper() for s in self.enabled_strategies)}")
        
        # Grid status
        if self.grids:
            print(f"\n📐 GRID TRADING:")
            total_profit = 0
            total_cycles = 0
            for symbol, grid in self.grids.items():
                s = grid.get_status()
                status_icon = "🟢" if s['active'] else "⚪"
                print(f"   {status_icon} {s['symbol']}: {s['range']} ({s['levels']} levels)")
                print(f"      Orders: {s['pending_buys']}B / {s['pending_sells']}S | "
                      f"Cycles: {s['completed']} | Profit: ${s['profit']:,.2f}")
                total_profit += s['profit']
                total_cycles += s['completed']
            
            roi = (total_profit / self.starting_capital * 100) if self.starting_capital > 0 else 0
            print(f"\n   💰 Grid Total: ${total_profit:,.2f} ({total_cycles} cycles, {roi:+.1f}% ROI)")
        
        # Future strategies add their status block here:
        # if self.some_new_strategy:
        #     print(f"\n NEW STRATEGY:")
        #     ...
        
        print("=" * 70)

    def run_once(self):
        """Run one iteration of all strategies."""
        # Check max drawdown
        if self.risk_manager.check_max_drawdown():
            self.alerts.error_alert(f"⛔ MAX DRAWDOWN EXCEEDED! Stopping bot.")
            self.running = False
            return
        
        self.update_regimes()        # <-- daily HMM regime check
        # Run strategies
        self.run_grid_strategy()
        self._check_daily_summary()
        
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
        
        # Check interval
        if interval_seconds is None:
            interval_seconds = 60  # 1 minute default
        
        try:
            while self.running:
                # Run trading logic
                self.run_once()
                
                # Print status
                self.print_status()
                
                # Check every interval_seconds
                wait_time = interval_seconds
                
                print(f"\n⏳ Next check in {wait_time/60:.1f} minutes...")
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n\n⛔ Bot stopped by user")
            self.alerts.status_alert("🛑 Bot stopped")
            for grid in self.grids.values():
                grid.shutdown()
        except Exception as e:
            self.alerts.error_alert(f"Bot crashed: {e}")
            traceback.print_exc()
            for grid in self.grids.values():
                grid.shutdown()
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
  grid      - Adaptive grid trading (active)

Examples:
  python live_trader.py                    # Paper trading
  python live_trader.py --live             # Real money
  python live_trader.py --strategies grid  # Grid only (default)
  python live_trader.py --capital 5000     # Custom capital
  python live_trader.py --interval 120     # Check every 2 minutes
  python live_trader.py --test             # One iteration, then exit
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
        choices=['grid'],
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
        help='Check interval in seconds (default: 60)'
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
