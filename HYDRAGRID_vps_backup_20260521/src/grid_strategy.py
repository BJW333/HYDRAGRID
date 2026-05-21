"""
Grid Trading Strategy
======================
Places buy orders below current price and sell orders above it.
Profits from natural price oscillation without predicting direction.

How it works:
    1. Define a price range and number of grid levels
    2. Place buy limit orders below price, sell limits above
    3. When a buy fills → immediately place a sell one level up
    4. When a sell fills → immediately place a buy one level down
    5. Each completed cycle captures the grid spacing as profit

Research shows grid bots produced 10-22% returns even in -50% markets
(CoinTelegraph Dec 2024-Apr 2025 backtest data).
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """Grid strategy parameters."""
    symbol: str = 'ETH/USD'
    lower_price: float = 0       # Auto-calculated if 0
    upper_price: float = 0       # Auto-calculated if 0
    num_grids: int = 15          # Number of grid levels
    total_investment: float = 0  # USD to allocate (set by live_trader)
    auto_range_pct: float = 0.08 # Auto-range: ±8% from current price
    adaptive: bool = True        # Auto-adjust range to volatility
    min_range_pct: float = 0.05  # Minimum range in low-vol (±5%)
    max_range_pct: float = 0.12  # Maximum range in high-vol (±12%)
    rebalance_hours: int = 336   # Recalibrate every 14 days (336 hours)
    
    # Safety
    stop_loss_pct: float = 0.15  # Kill grid if price drops 15% below range
    
    @property
    def grid_spacing(self) -> float:
        if self.upper_price <= self.lower_price:
            return 0
        return (self.upper_price - self.lower_price) / self.num_grids
    
    @property
    def profit_per_grid(self) -> float:
        """Estimated profit per completed grid cycle (before fees)."""
        if self.lower_price <= 0:
            return 0
        mid = (self.upper_price + self.lower_price) / 2
        return self.grid_spacing / mid  # As a percentage


@dataclass
class GridLevel:
    """A single price level in the grid."""
    price: float
    index: int
    order_id: Optional[str] = None
    order_side: Optional[str] = None  # 'buy' or 'sell'
    status: str = 'empty'  # 'empty', 'pending', 'filled'
    fill_price: float = 0


class GridStrategy:
    """
    Grid trading engine.
    
    Manages a grid of limit orders on a single trading pair.
    Designed to plug into LiveTrader's main loop.
    """
    
    def __init__(self, exchange, config: GridConfig, alerts=None):
        self.exchange = exchange
        self.config = config
        self.alerts = alerts
        
        self.levels: List[GridLevel] = []
        self.active = False
        self.completed_trades: int = 0
        self.total_profit: float = 0
        self.order_size_qty: float = 0  # Qty per grid order
        
        # State persistence
        self.state_file = Path(__file__).parent.parent / "data" / f"grid_state_{config.symbol.replace('/', '_')}.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
    def calculate_adaptive_range(self, candles_15m: list = None) -> float:
        """
        Calculate optimal grid range from recent ATR (Average True Range).
        Uses 15m candles to measure actual volatility, then sizes grid to fit.
        
        Returns optimal range_pct.
        """
        if not self.config.adaptive or not candles_15m or len(candles_15m) < 20:
            return self.config.auto_range_pct
        
        if isinstance(candles_15m, pd.DataFrame):
            df = candles_15m
        else:
            return self.config.auto_range_pct
        
        # Calculate ATR over last 96 bars (24 hours of 15m candles)
        lookback = min(96, len(df))
        recent = df.iloc[-lookback:]
        
        high = recent['high']
        low = recent['low']
        close = recent['close'].shift(1)
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close),
            'lc': abs(low - close),
        }).max(axis=1)
        
        atr = tr.mean()
        current_price = df.iloc[-1]['close']
        atr_pct = atr / current_price
        
        # Grid range = 6x ATR (covers ~6 hours of normal movement)
        # This ensures levels are spaced at roughly 1 ATR apart
        optimal_range = atr_pct * 4
        
        # Clamp to min/max
        optimal_range = max(self.config.min_range_pct, 
                          min(self.config.max_range_pct, optimal_range))
        
        logger.info(f"Adaptive grid {self.config.symbol}: ATR={atr_pct*100:.2f}% → range={optimal_range*100:.1f}%")
        
        return round(optimal_range, 4)
    
    # def needs_rebalance(self, current_price: float) -> bool:
    #     """Check if grid needs rebalancing (price near edge or timer expired)."""
    #     if not self.active or not self.config.adaptive:
    #         return False
        
    #     cfg = self.config
    #     range_size = cfg.upper_price - cfg.lower_price
    #     if range_size <= 0:
    #         return True
        
    #     # Rebalance if price is within 10% of grid edge
    #     lower_threshold = cfg.lower_price + range_size * 0.20
    #     upper_threshold = cfg.upper_price - range_size * 0.20
        
    #     if current_price < lower_threshold or current_price > upper_threshold:
    #         logger.info(f"Grid {cfg.symbol} rebalance triggered: price ${current_price:.2f} near edge")
    #         return True
        
    #     return False
    
    def needs_rebalance(self, current_price: float) -> bool:
        """Check if grid needs rebalancing (price near edge AND cooldown expired)."""
        if not self.active or not self.config.adaptive:
            return False
        
        # COOLDOWN: Don't rebalance more than once per rebalance_hours
        if hasattr(self, '_last_rebalance_time'):
            hours_since = (time.time() - self._last_rebalance_time) / 3600
            if hours_since < self.config.rebalance_hours:
                return False
        
        cfg = self.config
        range_size = cfg.upper_price - cfg.lower_price
        if range_size <= 0:
            return True
        
        # Rebalance if price is within 20% of grid edge
        lower_threshold = cfg.lower_price + range_size * 0.20
        upper_threshold = cfg.upper_price - range_size * 0.20
        
        if current_price < lower_threshold or current_price > upper_threshold:
            logger.info(f"Grid {cfg.symbol} rebalance triggered: price ${current_price:.2f} near edge")
            return True
        
        return False
    
    def rebalance(self, current_price: float, candles_15m=None):
        """Shut down current grid and set up fresh one with adaptive range."""
        self._last_rebalance_time = time.time()
        logger.info(f"♻️ Rebalancing grid {self.config.symbol} around ${current_price:.2f}")    
        
        # Cancel existing orders
        self.shutdown()
        
        # Calculate new adaptive range
        if candles_15m is not None:
            new_range = self.calculate_adaptive_range(candles_15m)
            self.config.auto_range_pct = new_range
        
        # Reset prices so setup() recalculates
        self.config.lower_price = 0
        self.config.upper_price = 0
        self.active = False
        
        # Re-setup
        self.setup(current_price)
    
    def setup(self, current_price: float):
        """
        Initialize the grid around the current price.
        Call this once at startup.
        """
        cfg = self.config
        
        # Auto-calculate range if not set
        if cfg.lower_price <= 0 or cfg.upper_price <= 0:
            cfg.lower_price = round(current_price * (1 - cfg.auto_range_pct), 2)
            cfg.upper_price = round(current_price * (1 + cfg.auto_range_pct), 2)
        
        # Calculate order size
        # Divide capital across buy-side grid levels (worst case: all buys fill)
        buy_levels = cfg.num_grids // 2
        if buy_levels > 0 and cfg.total_investment > 0:
            usd_per_level = cfg.total_investment / buy_levels
            mid_price = (cfg.upper_price + cfg.lower_price) / 2
            self.order_size_qty = round(usd_per_level / mid_price, 6)
        
        if self.order_size_qty <= 0:
            logger.error(f"Grid order size is 0 — check total_investment ({cfg.total_investment})")
            return
        
        # Create grid levels
        self.levels = []
        for i in range(cfg.num_grids + 1):
            price = round(cfg.lower_price + (i * cfg.grid_spacing), 2)
            self.levels.append(GridLevel(price=price, index=i))
        
        self.active = True
        
        spacing_pct = (cfg.grid_spacing / current_price) * 100
        msg = (
            f"📊 GRID SETUP: {cfg.symbol}\n"
            f"Range: ${cfg.lower_price:,.2f} - ${cfg.upper_price:,.2f}\n"
            f"Levels: {cfg.num_grids} | Spacing: ${cfg.grid_spacing:,.2f} ({spacing_pct:.2f}%)\n"
            f"Order size: {self.order_size_qty:.6f} (~${self.order_size_qty * current_price:,.2f})\n"
            f"Investment: ${cfg.total_investment:,.2f}"
        )
        logger.info(msg)
        if self.alerts:
            self.alerts.send(msg, level="INFO")
        
        # Place initial orders
        self._place_initial_orders(current_price)
        self._save_state()
    
    def _place_initial_orders(self, current_price: float):
        """Place buy orders below price, sell orders above."""
        placed = 0
        for level in self.levels:
            if abs(level.price - current_price) < self.config.grid_spacing * 0.3:
                continue  # Skip level too close to current price
            
            if level.price < current_price:
                # Buy order below current price
                self._place_order(level, 'buy')
                placed += 1
            elif level.price > current_price:
                # Sell order above current price (only if we have inventory)
                # For initial setup, we skip sell orders — they get placed after buys fill
                pass
        
        logger.info(f"Placed {placed} initial grid buy orders for {self.config.symbol}")
    
    def _place_order(self, level: GridLevel, side: str) -> bool:
        """Place a limit order at a grid level."""
        try:
            order = self.exchange.create_limit_order(
                symbol=self.config.symbol,
                side=side,
                amount=self.order_size_qty,
                price=level.price,
            )
            level.order_id = order['id']
            level.order_side = side
            level.status = 'pending'
            return True
        except Exception as e:
            logger.error(f"Grid order failed at ${level.price}: {e}")
            level.status = 'empty'
            return False
    
    def check_fills(self) -> List[dict]:
        """
        Check all pending orders for fills.
        Called every iteration of the main loop.
        Returns list of fill events.
        """
        if not self.active:
            return []
        
        fills = []
        
        for level in self.levels:
            if level.status != 'pending' or not level.order_id:
                continue
            
            try:
                order = self.exchange.get_order(level.order_id)
            except Exception as e:
                logger.debug(f"Order check failed for {level.order_id}: {e}")
                continue
            
            if order['status'] == 'filled':
                level.status = 'filled'
                level.fill_price = order['filled_avg_price'] or level.price
                
                fill_event = {
                    'side': level.order_side,
                    'price': level.fill_price,
                    'level_index': level.index,
                    'time': datetime.now().isoformat(),
                }
                fills.append(fill_event)
                
                # Place opposite order at adjacent level
                self._handle_fill(level)
        
        if fills:
            self._save_state()
        
        return fills
    
    def _handle_fill(self, filled_level: GridLevel):
        """After a fill, place the opposite order one level away."""
        if filled_level.order_side == 'buy':
            # Buy filled → place sell one level UP
            sell_index = filled_level.index + 1
            if sell_index < len(self.levels):
                target = self.levels[sell_index]
                if target.status == 'empty' or target.status == 'filled':
                    self._place_order(target, 'sell')
                    
                    profit = target.price - filled_level.price
                    # Profit is realized when the sell fills, not now
                    logger.info(
                        f"🟢 GRID BUY filled @ ${filled_level.price:,.2f} → "
                        f"SELL placed @ ${target.price:,.2f} "
                        f"(potential +${profit * self.order_size_qty:,.2f})"
                    )
                    if self.alerts:
                        self.alerts.send(
                            f"🟢 GRID BUY {self.config.symbol} @ ${filled_level.price:,.2f}\n"
                            f"Sell queued @ ${target.price:,.2f}",
                            level="TRADE"
                        )
            
            # Also re-place buy at this level for next dip
            filled_level.status = 'empty'
            filled_level.order_id = None
            
        elif filled_level.order_side == 'sell':
            # Sell filled → record profit, place buy one level DOWN
            buy_index = filled_level.index - 1
            
            # Record completed round-trip profit
            profit_per_unit = self.config.grid_spacing
            trade_profit = profit_per_unit * self.order_size_qty
            # Subtract fees (both sides)
            fee = filled_level.fill_price * self.order_size_qty * 0.003  # ~0.3% round trip
            trade_profit -= fee
            
            self.completed_trades += 1
            self.total_profit += trade_profit
            
            logger.info(
                f"✅ GRID SELL filled @ ${filled_level.price:,.2f} | "
                f"Profit: ${trade_profit:,.2f} | "
                f"Total: ${self.total_profit:,.2f} ({self.completed_trades} cycles)"
            )
            if self.alerts:
                self.alerts.send(
                    f"✅ GRID SELL {self.config.symbol} @ ${filled_level.price:,.2f}\n"
                    f"Cycle profit: ${trade_profit:,.2f} | "
                    f"Total: ${self.total_profit:,.2f} ({self.completed_trades} cycles)",
                    level="TRADE"
                )
            
            if buy_index >= 0:
                target = self.levels[buy_index]
                if target.status == 'empty' or target.status == 'filled':
                    self._place_order(target, 'buy')
            
            filled_level.status = 'empty'
            filled_level.order_id = None
    
    def check_safety(self, current_price: float) -> bool:
        """
        Check if price has moved outside the grid range.
        Returns True if grid should be stopped.
        """
        if not self.active:
            return False
        
        stop_price = self.config.lower_price * (1 - self.config.stop_loss_pct)
        if current_price < stop_price:
            logger.warning(f"⛔ Grid stop-loss triggered: ${current_price} < ${stop_price}")
            self.shutdown()
            return True
        
        return False
    
    def shutdown(self):
        """Cancel all pending grid orders and deactivate."""
        self.active = False
        cancelled = 0
        for level in self.levels:
            if level.status == 'pending' and level.order_id:
                try:
                    self.exchange.cancel_order(level.order_id)
                    cancelled += 1
                except Exception:
                    pass
                level.status = 'empty'
                level.order_id = None
        
        msg = (
            f"🛑 GRID SHUTDOWN: {self.config.symbol}\n"
            f"Cancelled {cancelled} orders\n"
            f"Total profit: ${self.total_profit:,.2f} ({self.completed_trades} cycles)"
        )
        logger.info(msg)
        if self.alerts:
            self.alerts.send(msg, level="INFO")
        self._save_state()
    
    def get_status(self) -> dict:
        """Get current grid status for display."""
        pending_buys = sum(1 for l in self.levels if l.status == 'pending' and l.order_side == 'buy')
        pending_sells = sum(1 for l in self.levels if l.status == 'pending' and l.order_side == 'sell')
        return {
            'symbol': self.config.symbol,
            'active': self.active,
            'range': f"${self.config.lower_price:,.2f} - ${self.config.upper_price:,.2f}",
            'levels': self.config.num_grids,
            'spacing': f"${self.config.grid_spacing:,.2f}",
            'pending_buys': pending_buys,
            'pending_sells': pending_sells,
            'completed': self.completed_trades,
            'profit': self.total_profit,
        }
    
    def _save_state(self):
        """Persist grid state."""
        state = {
            'config': {
                'symbol': self.config.symbol,
                'lower_price': self.config.lower_price,
                'upper_price': self.config.upper_price,
                'num_grids': self.config.num_grids,
                'total_investment': self.config.total_investment,
            },
            'order_size_qty': self.order_size_qty,
            'completed_trades': self.completed_trades,
            'total_profit': self.total_profit,
            'active': self.active,
            'levels': [
                {
                    'price': l.price,
                    'index': l.index,
                    'order_id': l.order_id,
                    'order_side': l.order_side,
                    'status': l.status,
                }
                for l in self.levels
            ],
            'last_update': datetime.now().isoformat(),
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> bool:
        """Load persisted grid state. Returns True if state was loaded."""
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            cfg = state.get('config', {})
            self.config.symbol = cfg.get('symbol', self.config.symbol)
            self.config.lower_price = cfg.get('lower_price', 0)
            self.config.upper_price = cfg.get('upper_price', 0)
            self.config.num_grids = cfg.get('num_grids', self.config.num_grids)
            self.config.total_investment = cfg.get('total_investment', 0)
            
            self.order_size_qty = state.get('order_size_qty', 0)
            self.completed_trades = state.get('completed_trades', 0)
            self.total_profit = state.get('total_profit', 0)
            self.active = state.get('active', False)
            
            self.levels = []
            for ld in state.get('levels', []):
                level = GridLevel(
                    price=ld['price'],
                    index=ld['index'],
                    order_id=ld.get('order_id'),
                    order_side=ld.get('order_side'),
                    status=ld.get('status', 'empty'),
                )
                self.levels.append(level)
            
            if self.active:
                logger.info(f"Restored grid: {self.config.symbol} with {len(self.levels)} levels")
            return self.active
            
        except Exception as e:
            logger.error(f"Failed to load grid state: {e}")
            return False
