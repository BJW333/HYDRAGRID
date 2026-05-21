"""
Risk Manager Module
===================
Handles position sizing, stop losses, profit targets, and trade management.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    STOP_LOSS_ATR, PROFIT_TARGET_ATR, BREAKEVEN_TRIGGER_ATR,
    TRAILING_STOP_ATR, RISK_PER_TRADE, MAX_DRAWDOWN
)


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1


class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"
    TRAILING_STOP = "trailing_stop"
    BREAKEVEN_STOP = "breakeven_stop"
    MANUAL = "manual"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass
class TradePosition:
    """Represents an open trade position."""
    
    entry_price: float
    entry_time: pd.Timestamp
    direction: TradeDirection
    size: float
    stop_loss: float
    profit_target: float
    breakeven_trigger: float
    atr_at_entry: float
    
    # State tracking
    stop_moved_to_breakeven: bool = False
    highest_price: float = field(default=None)
    lowest_price: float = field(default=None)
    current_stop: float = field(default=None)
    
    def __post_init__(self):
        """Initialize tracking values."""
        if self.highest_price is None:
            self.highest_price = self.entry_price
        if self.lowest_price is None:
            self.lowest_price = self.entry_price
        if self.current_stop is None:
            self.current_stop = self.stop_loss
    
    def update_price_extremes(self, high: float, low: float):
        """Update highest/lowest price reached."""
        self.highest_price = max(self.highest_price, high)
        self.lowest_price = min(self.lowest_price, low)
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.direction == TradeDirection.LONG:
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size
    
    def unrealized_pnl_percent(self, current_price: float) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.direction == TradeDirection.LONG:
            return (current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - current_price) / self.entry_price * 100
    
    def r_multiple(self, current_price: float) -> float:
        """Calculate R-multiple (profit relative to initial risk)."""
        initial_risk = abs(self.entry_price - self.stop_loss)
        if initial_risk == 0:
            return 0
        
        if self.direction == TradeDirection.LONG:
            profit = current_price - self.entry_price
        else:
            profit = self.entry_price - current_price
        
        return profit / initial_risk


class RiskManager:
    """
    Manages position sizing and trade exits.
    
    Responsible for:
    - Calculating position sizes based on risk
    - Setting stop losses and profit targets
    - Managing trailing stops
    - Tracking account drawdown
    """
    
    def __init__(self, initial_capital: float, risk_per_trade: float = RISK_PER_TRADE,
                 max_drawdown: float = MAX_DRAWDOWN):
        """
        Initialize the risk manager.
        
        Args:
            initial_capital: Starting account balance
            risk_per_trade: Fraction of account to risk per trade (e.g., 0.02 = 2%)
            max_drawdown: Maximum allowed drawdown before stopping (e.g., 0.15 = 15%)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        
    def update_capital(self, new_capital: float):
        """
        Update current capital after a trade.
        
        Args:
            new_capital: New account balance
        """
        self.current_capital = new_capital
        
        # Update peak and drawdown
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        
        self.current_drawdown = (self.peak_capital - new_capital) / self.peak_capital
    
    def check_max_drawdown(self) -> bool:
        """
        Check if max drawdown has been exceeded.
        
        Returns:
            True if trading should stop
        """
        return self.current_drawdown >= self.max_drawdown
    
    def calculate_position_size(self, entry_price: float, stop_price: float,
                                account_override: float = None) -> float:
        """
        Calculate position size based on risk.
        
        Args:
            entry_price: Planned entry price
            stop_price: Stop loss price
            account_override: Optional account size override
            
        Returns:
            Number of units to trade
        """
        account = account_override or self.current_capital
        
        # Amount willing to risk
        risk_amount = account * self.risk_per_trade
        
        # Risk per unit
        risk_per_unit = abs(entry_price - stop_price)
        
        if risk_per_unit == 0:
            return 0
        
        # Position size
        position_size = risk_amount / risk_per_unit
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, atr: float,
                            direction: TradeDirection,
                            atr_multiplier: float = STOP_LOSS_ATR) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            direction: Trade direction
            atr_multiplier: Number of ATRs for stop distance
            
        Returns:
            Stop loss price
        """
        stop_distance = atr * atr_multiplier
        
        if direction == TradeDirection.LONG:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_profit_target(self, entry_price: float, atr: float,
                                direction: TradeDirection,
                                atr_multiplier: float = PROFIT_TARGET_ATR) -> float:
        """
        Calculate profit target price.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            direction: Trade direction
            atr_multiplier: Number of ATRs for target distance
            
        Returns:
            Profit target price
        """
        target_distance = atr * atr_multiplier
        
        if direction == TradeDirection.LONG:
            return entry_price + target_distance
        else:
            return entry_price - target_distance
    
    def calculate_breakeven_trigger(self, entry_price: float, atr: float,
                                    direction: TradeDirection,
                                    atr_multiplier: float = BREAKEVEN_TRIGGER_ATR) -> float:
        """
        Calculate price at which to move stop to breakeven.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            direction: Trade direction
            atr_multiplier: Number of ATRs for trigger distance
            
        Returns:
            Breakeven trigger price
        """
        trigger_distance = atr * atr_multiplier
        
        if direction == TradeDirection.LONG:
            return entry_price + trigger_distance
        else:
            return entry_price - trigger_distance
    
    def create_position(self, entry_price: float, entry_time: pd.Timestamp,
                        atr: float, direction: TradeDirection) -> TradePosition:
        """
        Create a new trade position with all levels calculated.
        
        Args:
            entry_price: Entry price
            entry_time: Entry timestamp
            atr: Current ATR value
            direction: Trade direction
            
        Returns:
            TradePosition object
        """
        stop_loss = self.calculate_stop_loss(entry_price, atr, direction)
        profit_target = self.calculate_profit_target(entry_price, atr, direction)
        breakeven_trigger = self.calculate_breakeven_trigger(entry_price, atr, direction)
        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        return TradePosition(
            entry_price=entry_price,
            entry_time=entry_time,
            direction=direction,
            size=position_size,
            stop_loss=stop_loss,
            profit_target=profit_target,
            breakeven_trigger=breakeven_trigger,
            atr_at_entry=atr
        )
    
    def check_exit(self, position: TradePosition, high: float, low: float,
                   close: float) -> Tuple[bool, Optional[ExitReason], Optional[float]]:
        """
        Check if position should be exited.
        
        Args:
            position: Current position
            high: Candle high
            low: Candle low
            close: Candle close
            
        Returns:
            Tuple of (should_exit, exit_reason, exit_price)
        """
        # Update price extremes
        position.update_price_extremes(high, low)
        
        if position.direction == TradeDirection.LONG:
            # Check stop loss
            if low <= position.current_stop:
                return True, ExitReason.STOP_LOSS, position.current_stop
            
            # Check profit target
            if high >= position.profit_target:
                return True, ExitReason.PROFIT_TARGET, position.profit_target
            
            # Check breakeven trigger (move stop if not already done)
            if not position.stop_moved_to_breakeven:
                if high >= position.breakeven_trigger:
                    position.current_stop = position.entry_price
                    position.stop_moved_to_breakeven = True
        
        else:  # SHORT
            # Check stop loss
            if high >= position.current_stop:
                return True, ExitReason.STOP_LOSS, position.current_stop
            
            # Check profit target
            if low <= position.profit_target:
                return True, ExitReason.PROFIT_TARGET, position.profit_target
            
            # Check breakeven trigger
            if not position.stop_moved_to_breakeven:
                if low <= position.breakeven_trigger:
                    position.current_stop = position.entry_price
                    position.stop_moved_to_breakeven = True
        
        return False, None, None
    
    def update_trailing_stop(self, position: TradePosition,
                             trailing_atr_mult: float = TRAILING_STOP_ATR) -> float:
        """
        Update trailing stop based on price movement.
        
        Only trails in profitable direction, never moves stop backwards.
        
        Args:
            position: Current position
            trailing_atr_mult: ATR multiplier for trailing distance
            
        Returns:
            New stop price
        """
        if not position.stop_moved_to_breakeven:
            return position.current_stop
        
        trail_distance = position.atr_at_entry * trailing_atr_mult
        
        if position.direction == TradeDirection.LONG:
            new_stop = position.highest_price - trail_distance
            position.current_stop = max(position.current_stop, new_stop)
        else:
            new_stop = position.lowest_price + trail_distance
            position.current_stop = min(position.current_stop, new_stop)
        
        return position.current_stop
    
    def get_risk_metrics(self) -> dict:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': self.current_drawdown * 100,  # As percentage
            'max_drawdown_limit': self.max_drawdown * 100,
            'risk_per_trade': self.risk_per_trade * 100,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            'drawdown_exceeded': self.check_max_drawdown()
        }


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_reward_risk_ratio(entry: float, stop: float, target: float) -> float:
    """
    Calculate reward to risk ratio.
    
    Args:
        entry: Entry price
        stop: Stop loss price
        target: Profit target price
        
    Returns:
        Reward to risk ratio
    """
    risk = abs(entry - stop)
    reward = abs(target - entry)
    
    if risk == 0:
        return 0
    
    return reward / risk


def estimate_position_value(entry_price: float, position_size: float) -> float:
    """
    Calculate total position value.
    
    Args:
        entry_price: Entry price
        position_size: Number of units
        
    Returns:
        Total position value
    """
    return entry_price * position_size


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Risk Manager Module...")
    print("=" * 50)
    
    # Initialize risk manager
    rm = RiskManager(initial_capital=10000, risk_per_trade=0.02)
    
    # Test position creation
    print("\nCreating test position:")
    print("  Entry: $3,000")
    print("  ATR: $50")
    print("  Direction: LONG")
    
    position = rm.create_position(
        entry_price=3000,
        entry_time=pd.Timestamp.now(),
        atr=50,
        direction=TradeDirection.LONG
    )
    
    print(f"\nPosition Details:")
    print(f"  Size: {position.size:.4f} units")
    print(f"  Stop Loss: ${position.stop_loss:.2f}")
    print(f"  Profit Target: ${position.profit_target:.2f}")
    print(f"  Breakeven Trigger: ${position.breakeven_trigger:.2f}")
    
    # Calculate R:R
    rr = calculate_reward_risk_ratio(
        position.entry_price,
        position.stop_loss,
        position.profit_target
    )
    print(f"  Reward:Risk Ratio: {rr:.2f}")
    
    # Test position value
    value = estimate_position_value(position.entry_price, position.size)
    print(f"  Position Value: ${value:.2f}")
    
    # Test exit checks
    print("\nTesting exit scenarios:")
    
    # Scenario 1: Price hits breakeven trigger
    should_exit, reason, price = rm.check_exit(position, high=3060, low=2980, close=3050)
    print(f"  Candle 1 (H:3060 L:2980 C:3050): Exit={should_exit}, Stop moved to BE: {position.stop_moved_to_breakeven}")
    
    # Scenario 2: Price continues up
    should_exit, reason, price = rm.check_exit(position, high=3100, low=3040, close=3090)
    print(f"  Candle 2 (H:3100 L:3040 C:3090): Exit={should_exit}")
    
    # Scenario 3: Price drops to breakeven stop
    should_exit, reason, price = rm.check_exit(position, high=3020, low=2990, close=3000)
    print(f"  Candle 3 (H:3020 L:2990 C:3000): Exit={should_exit}, Reason={reason}, Price=${price}")
    
    # Test risk metrics
    print("\nRisk Metrics:")
    metrics = rm.get_risk_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
