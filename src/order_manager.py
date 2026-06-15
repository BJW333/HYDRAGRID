#!/usr/bin/env python3
"""
Order Manager — Centralized order routing and tracking
=======================================================
All orders go through here. Strategies decide WHAT to trade.
OrderManager handles HOW and WHERE.

Supports:
  - Multiple exchanges (split orders to reduce market impact)
  - Limit orders (grid strategy)
  - Market orders (future directional strategies)
  - Unified P&L tracking across all strategies
  - Order state persistence
  - Fill detection across all exchanges

Usage:
    # In live_trader.py:
    from src.order_manager import OrderManager
    
    om = OrderManager()
    om.add_exchange('alpaca', alpaca_adapter)
    om.add_exchange('coinbase', coinbase_adapter)  # Optional
    
    # Grid strategy calls:
    order_id = om.place_limit_buy('SOL/USD', price=83.50, qty=0.78, strategy='GRID')
    order_id = om.place_limit_sell('SOL/USD', price=84.50, qty=0.78, strategy='GRID')
    fills = om.check_fills('SOL/USD')
    om.cancel_order(order_id)
    
    # Future directional strategy calls:
    order_id = om.place_market_buy('ETH/USD', qty=0.5, strategy='BREAKOUT')
    order_id = om.place_market_sell('ETH/USD', qty=0.5, strategy='BREAKOUT')
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Tracks a single order across its lifecycle."""
    order_id: str               # Exchange order ID
    internal_id: str            # Our tracking ID (om_001, om_002, ...)
    symbol: str
    side: str                   # 'buy' or 'sell'
    order_type: str             # 'limit' or 'market'
    price: float                # Limit price (0 for market)
    qty: float
    strategy: str               # 'GRID', 'BREAKOUT', etc.
    exchange: str               # Which exchange this is on
    status: str = 'pending'     # 'pending', 'filled', 'cancelled', 'failed'
    fill_price: float = 0.0     # Actual fill price
    fill_time: str = ''
    created_at: str = ''
    meta: dict = field(default_factory=dict)  # Strategy-specific data (grid_level, etc.)
    
    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class Fill:
    """A completed fill event."""
    internal_id: str
    order_id: str
    symbol: str
    side: str
    price: float
    qty: float
    strategy: str
    exchange: str
    time: str
    fee: float = 0.0
    meta: dict = field(default_factory=dict)


class OrderManager:
    """
    Centralized order management.
    
    All strategies place orders through here.
    Supports multiple exchanges with automatic order routing.
    Tracks all orders and P&L in one place.
    """
    
    def __init__(self, fee_rate: float = 0.0015, state_dir: str = None):
        """
        Args:
            fee_rate: Default fee rate for P&L calculations
            state_dir: Directory to save order state (default: data/)
        """
        self.exchanges: Dict[str, object] = {}  # name -> exchange adapter
        self.orders: Dict[str, Order] = {}       # internal_id -> Order
        self.fills: List[Fill] = []
        self.fee_rate = fee_rate
        
        # P&L tracking per strategy
        self.pnl: Dict[str, float] = {}          # strategy -> total P&L
        self.trade_count: Dict[str, int] = {}     # strategy -> completed trades
        
        # Internal ID counter
        self._next_id = 1
        
        # State persistence
        if state_dir:
            self.state_file = Path(state_dir) / "order_manager_state.json"
        else:
            self.state_file = Path(__file__).parent.parent / "data" / "order_manager_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_state()
    
    # =========================================================================
    # EXCHANGE MANAGEMENT
    # =========================================================================
    
    def add_exchange(self, name: str, adapter):
        """
        Register an exchange adapter.
        
        Args:
            name: Identifier (e.g., 'alpaca', 'coinbase', 'kraken')
            adapter: Exchange object with create_limit_order(), create_market_order(),
                     get_order(), cancel_order() methods
        """
        self.exchanges[name] = adapter
        logger.info(f"OrderManager: registered exchange '{name}'")
    
    def get_exchange(self, name: str = None):
        """Get an exchange adapter. If name is None, returns the first/default."""
        if name:
            return self.exchanges.get(name)
        if self.exchanges:
            return next(iter(self.exchanges.values()))
        return None
    
    @property
    def num_exchanges(self):
        return len(self.exchanges)
    
    def _pick_exchange(self, symbol: str = None) -> tuple:
        """
        Pick which exchange to route an order to.
        Simple round-robin for now. Can be upgraded to smart routing later.
        
        Returns:
            (exchange_name, exchange_adapter)
        """
        if not self.exchanges:
            raise RuntimeError("No exchanges registered. Call add_exchange() first.")
        
        # For now: use the first exchange
        # TODO: Smart routing based on order book depth, fees, fill rates
        name = next(iter(self.exchanges))
        return name, self.exchanges[name]
    
    def _pick_exchanges_for_split(self, symbol: str, qty: float) -> List[tuple]:
        """
        Split an order across multiple exchanges to reduce market impact.
        
        Returns:
            List of (exchange_name, exchange_adapter, split_qty)
        """
        if len(self.exchanges) <= 1:
            name, ex = self._pick_exchange(symbol)
            return [(name, ex, qty)]
        
        # Even split across all exchanges
        split_qty = qty / len(self.exchanges)
        return [(name, ex, split_qty) for name, ex in self.exchanges.items()]
    
    def _gen_id(self) -> str:
        """Generate unique internal order ID."""
        oid = f"om_{self._next_id:06d}"
        self._next_id += 1
        return oid
    
    # =========================================================================
    # LIMIT ORDERS (used by grid)
    # =========================================================================
    
    def place_limit_buy(self, symbol: str, price: float, qty: float,
                        strategy: str = 'GRID', exchange_name: str = None,
                        meta: dict = None) -> Optional[str]:
        """
        Place a limit buy order.
        
        Args:
            symbol: Trading pair (e.g., 'SOL/USD')
            price: Limit price
            qty: Quantity to buy
            strategy: Strategy that requested this order
            exchange_name: Specific exchange (None = auto-pick)
            meta: Strategy-specific metadata (e.g., grid_level_index)
        
        Returns:
            Internal order ID, or None if failed
        """
        return self._place_limit(symbol, 'buy', price, qty, strategy, exchange_name, meta)
    
    def place_limit_sell(self, symbol: str, price: float, qty: float,
                         strategy: str = 'GRID', exchange_name: str = None,
                         meta: dict = None) -> Optional[str]:
        """Place a limit sell order."""
        return self._place_limit(symbol, 'sell', price, qty, strategy, exchange_name, meta)
    
    def _place_limit(self, symbol, side, price, qty, strategy, exchange_name, meta) -> Optional[str]:
        """Internal: place a limit order on a specific or auto-picked exchange."""
        try:
            if exchange_name and exchange_name in self.exchanges:
                ex_name, ex = exchange_name, self.exchanges[exchange_name]
            else:
                ex_name, ex = self._pick_exchange(symbol)
            
            order = ex.create_limit_order(
                symbol=symbol,
                side=side,
                amount=qty,
                price=price,
            )
            
            internal_id = self._gen_id()
            self.orders[internal_id] = Order(
                order_id=order['id'],
                internal_id=internal_id,
                symbol=symbol,
                side=side,
                order_type='limit',
                price=price,
                qty=qty,
                strategy=strategy,
                exchange=ex_name,
                status='pending',
                created_at=datetime.now().isoformat(),
                meta=meta or {},
            )
            
            logger.debug(f"OM: {side.upper()} limit {symbol} {qty} @ ${price:,.2f} on {ex_name} [{internal_id}]")
            return internal_id
            
        except Exception as e:
            logger.error(f"OM: limit {side} failed for {symbol}: {e}")
            return None
    
    # =========================================================================
    # MARKET ORDERS (used by directional strategies)
    # =========================================================================
    
    def place_market_buy(self, symbol: str, qty: float, strategy: str = 'GRID',
                         split: bool = False, meta: dict = None) -> Optional[str]:
        """
        Place a market buy order.
        
        Args:
            split: If True, split order across all exchanges
        """
        return self._place_market(symbol, 'buy', qty, strategy, split, meta)
    
    def place_market_sell(self, symbol: str, qty: float, strategy: str = 'GRID',
                          split: bool = False, meta: dict = None) -> Optional[str]:
        """Place a market sell order."""
        return self._place_market(symbol, 'sell', qty, strategy, split, meta)
    
    def _place_market(self, symbol, side, qty, strategy, split, meta) -> Optional[str]:
        """Internal: place market order, optionally split across exchanges."""
        try:
            if split and len(self.exchanges) > 1:
                # Split across exchanges — return the first internal_id
                first_id = None
                for ex_name, ex, split_qty in self._pick_exchanges_for_split(symbol, qty):
                    order = ex.create_market_order(symbol=symbol, side=side, amount=split_qty)
                    internal_id = self._gen_id()
                    self.orders[internal_id] = Order(
                        order_id=order['id'],
                        internal_id=internal_id,
                        symbol=symbol,
                        side=side,
                        order_type='market',
                        price=0,
                        qty=split_qty,
                        strategy=strategy,
                        exchange=ex_name,
                        status='filled',  # Market orders fill immediately
                        fill_price=order.get('average', order.get('price', 0)),
                        fill_time=datetime.now().isoformat(),
                        created_at=datetime.now().isoformat(),
                        meta=meta or {},
                    )
                    if first_id is None:
                        first_id = internal_id
                return first_id
            else:
                ex_name, ex = self._pick_exchange(symbol)
                order = ex.create_market_order(symbol=symbol, side=side, amount=qty)
                
                internal_id = self._gen_id()
                self.orders[internal_id] = Order(
                    order_id=order['id'],
                    internal_id=internal_id,
                    symbol=symbol,
                    side=side,
                    order_type='market',
                    price=0,
                    qty=qty,
                    strategy=strategy,
                    exchange=ex_name,
                    status='filled',
                    fill_price=order.get('average', order.get('price', 0)),
                    fill_time=datetime.now().isoformat(),
                    created_at=datetime.now().isoformat(),
                    meta=meta or {},
                )
                
                logger.info(f"OM: {side.upper()} market {symbol} {qty} on {ex_name} [{internal_id}]")
                return internal_id
                
        except Exception as e:
            logger.error(f"OM: market {side} failed for {symbol}: {e}")
            return None
    
    # =========================================================================
    # ORDER CHECKING & FILLS
    # =========================================================================
    
    def check_fills(self, symbol: str = None, strategy: str = None) -> List[Fill]:
        """
        Check all pending orders for fills.
        
        Args:
            symbol: Filter by symbol (None = check all)
            strategy: Filter by strategy (None = check all)
        
        Returns:
            List of new Fill events since last check
        """
        new_fills = []
        
        for internal_id, order in list(self.orders.items()):
            if order.status != 'pending':
                continue
            if symbol and order.symbol != symbol:
                continue
            if strategy and order.strategy != strategy:
                continue
            
            # Get the exchange this order is on
            ex = self.exchanges.get(order.exchange)
            if not ex:
                continue
            
            try:
                status = ex.get_order(order.order_id)
            except Exception as e:
                logger.debug(f"OM: order check failed {order.order_id}: {e}")
                continue
            
            if status['status'] == 'filled':
                order.status = 'filled'
                order.fill_price = status.get('filled_avg_price') or order.price
                order.fill_time = datetime.now().isoformat()
                
                # Calculate fee
                fee = order.fill_price * order.qty * self.fee_rate
                
                fill = Fill(
                    internal_id=internal_id,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    price=order.fill_price,
                    qty=order.qty,
                    strategy=order.strategy,
                    exchange=order.exchange,
                    time=order.fill_time,
                    fee=fee,
                    meta=order.meta,
                )
                
                new_fills.append(fill)
                self.fills.append(fill)
                
                logger.info(
                    f"OM: FILL {order.side.upper()} {order.symbol} "
                    f"{order.qty} @ ${order.fill_price:,.2f} on {order.exchange} "
                    f"[{order.strategy}]"
                )
            
            elif status['status'] in ['cancelled', 'canceled', 'expired', 'rejected']:
                order.status = 'cancelled'
        
        if new_fills:
            self._save_state()
        
        return new_fills
    
    def get_order(self, order_id: str) -> dict:
        """
        Get order status. Drop-in replacement for exchange.get_order().
        Accepts EITHER internal_id (om_000001) or exchange order ID.
        Returns dict matching Alpaca's format so grid works unchanged.
        """
        # Try internal ID first
        order = self.orders.get(order_id)
        if not order:
            # Try exchange order ID
            order = self.get_order_by_exchange_id(order_id)
        
        if order:
            # If still pending, check with exchange for latest status
            if order.status == 'pending':
                ex = self.exchanges.get(order.exchange)
                if ex:
                    try:
                        status = ex.get_order(order.order_id)
                        if status['status'] == 'filled':
                            order.status = 'filled'
                            order.fill_price = status.get('filled_avg_price') or order.price
                            order.fill_time = datetime.now().isoformat()
                        elif status['status'] in ['cancelled', 'canceled', 'expired', 'rejected']:
                            order.status = 'cancelled'
                    except Exception:
                        pass
            
            return {
                'id': order.order_id,
                'internal_id': order.internal_id,
                'status': order.status,
                'filled_qty': order.qty if order.status == 'filled' else 0,
                'filled_avg_price': order.fill_price,
                'side': order.side,
                'info': {},
            }
        
        # Fallback: check exchanges directly (order not tracked by OM)
        for ex in self.exchanges.values():
            try:
                return ex.get_order(order_id)
            except Exception:
                continue
        
        return {'id': order_id, 'status': 'unknown', 'filled_avg_price': 0, 'info': {}}
    
    def get_order_by_exchange_id(self, exchange_order_id: str) -> Optional[Order]:
        """Find an order by its exchange-side order ID."""
        for order in self.orders.values():
            if order.order_id == exchange_order_id:
                return order
        return None
    
    def get_pending_orders(self, symbol: str = None, strategy: str = None) -> List[Order]:
        """Get all pending orders, optionally filtered."""
        return [
            o for o in self.orders.values()
            if o.status == 'pending'
            and (symbol is None or o.symbol == symbol)
            and (strategy is None or o.strategy == strategy)
        ]
    
    # =========================================================================
    # CANCEL
    # =========================================================================
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order. Accepts EITHER internal_id or exchange order ID.
        Drop-in replacement for exchange.cancel_order().
        """
        # Try internal ID first
        order = self.orders.get(order_id)
        if not order:
            # Try exchange order ID
            order = self.get_order_by_exchange_id(order_id)
        
        if order and order.status == 'pending':
            ex = self.exchanges.get(order.exchange)
            if ex:
                try:
                    ex.cancel_order(order.order_id)
                    order.status = 'cancelled'
                    self._save_state()
                    return True
                except Exception as e:
                    logger.error(f"OM: cancel failed {order_id}: {e}")
                    return False
        
        # Fallback: try cancelling directly on all exchanges
        for ex in self.exchanges.values():
            try:
                ex.cancel_order(order_id)
                return True
            except Exception:
                continue
        return False
    
    def cancel_order_by_exchange_id(self, exchange_order_id: str) -> bool:
        """Cancel by exchange order ID (for backward compatibility with grid)."""
        order = self.get_order_by_exchange_id(exchange_order_id)
        if order:
            return self.cancel_order(order.internal_id)
        
        # Fallback: try cancelling directly on all exchanges
        for ex in self.exchanges.values():
            try:
                ex.cancel_order(exchange_order_id)
                return True
            except Exception:
                continue
        return False
    
    def cancel_all(self, symbol: str = None, strategy: str = None) -> int:
        """Cancel all pending orders, optionally filtered. Returns count cancelled."""
        cancelled = 0
        for order in self.get_pending_orders(symbol, strategy):
            if self.cancel_order(order.internal_id):
                cancelled += 1
        return cancelled
    
    # =========================================================================
    # P&L TRACKING
    # =========================================================================
    
    def record_trade_pnl(self, strategy: str, pnl: float):
        """Record a completed trade's P&L."""
        self.pnl[strategy] = self.pnl.get(strategy, 0) + pnl
        self.trade_count[strategy] = self.trade_count.get(strategy, 0) + 1
        self._save_state()
    
    def get_pnl(self, strategy: str = None) -> float:
        """Get total P&L, optionally for a specific strategy."""
        if strategy:
            return self.pnl.get(strategy, 0)
        return sum(self.pnl.values())
    
    def get_trade_count(self, strategy: str = None) -> int:
        """Get trade count, optionally for a specific strategy."""
        if strategy:
            return self.trade_count.get(strategy, 0)
        return sum(self.trade_count.values())
    
    def get_stats(self) -> dict:
        """Get comprehensive order manager stats."""
        pending = [o for o in self.orders.values() if o.status == 'pending']
        filled = [o for o in self.orders.values() if o.status == 'filled']
        
        # P&L by strategy
        strategy_stats = {}
        for strat in set(list(self.pnl.keys()) + [o.strategy for o in self.orders.values()]):
            strat_orders = [o for o in self.orders.values() if o.strategy == strat]
            strategy_stats[strat] = {
                'pnl': self.pnl.get(strat, 0),
                'trades': self.trade_count.get(strat, 0),
                'pending_orders': len([o for o in strat_orders if o.status == 'pending']),
                'filled_orders': len([o for o in strat_orders if o.status == 'filled']),
            }
        
        # P&L by exchange
        exchange_stats = {}
        for ex_name in self.exchanges:
            ex_fills = [f for f in self.fills if f.exchange == ex_name]
            exchange_stats[ex_name] = {
                'fills': len(ex_fills),
                'total_fees': sum(f.fee for f in ex_fills),
                'volume': sum(f.price * f.qty for f in ex_fills),
            }
        
        return {
            'exchanges': list(self.exchanges.keys()),
            'total_orders': len(self.orders),
            'pending': len(pending),
            'filled': len(filled),
            'total_fills': len(self.fills),
            'total_pnl': self.get_pnl(),
            'total_trades': self.get_trade_count(),
            'total_fees': sum(f.fee for f in self.fills),
            'by_strategy': strategy_stats,
            'by_exchange': exchange_stats,
        }
    
    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================
    
    def _save_state(self):
        """Save order state to disk."""
        state = {
            'next_id': self._next_id,
            'pnl': self.pnl,
            'trade_count': self.trade_count,
            # Only save recent orders (last 500) to keep file manageable
            'orders': {
                k: v.to_dict() for k, v in list(self.orders.items())[-500:]
            },
            'last_update': datetime.now().isoformat(),
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"OM: failed to save state: {e}")
    
    def _load_state(self):
        """Load order state from disk."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self._next_id = state.get('next_id', 1)
            self.pnl = state.get('pnl', {})
            self.trade_count = state.get('trade_count', {})
            
            # Restore orders
            for k, v in state.get('orders', {}).items():
                self.orders[k] = Order(
                    order_id=v['order_id'],
                    internal_id=v['internal_id'],
                    symbol=v['symbol'],
                    side=v['side'],
                    order_type=v['order_type'],
                    price=v['price'],
                    qty=v['qty'],
                    strategy=v['strategy'],
                    exchange=v['exchange'],
                    status=v['status'],
                    fill_price=v.get('fill_price', 0),
                    fill_time=v.get('fill_time', ''),
                    created_at=v.get('created_at', ''),
                    meta=v.get('meta', {}),
                )
            
            logger.info(f"OM: restored {len(self.orders)} orders, P&L: ${self.get_pnl():,.2f}")
            
        except Exception as e:
            logger.error(f"OM: failed to load state: {e}")
    
    # =========================================================================
    # CONVENIENCE — backward compatible with grid_strategy's exchange calls
    # =========================================================================
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float,
                           strategy: str = 'GRID', meta: dict = None) -> dict:
        """
        Drop-in replacement for exchange.create_limit_order().
        Grid strategy can use this without changing its internal logic.
        
        Returns dict with 'id' key (internal_id) for compatibility.
        """
        if side == 'buy':
            internal_id = self.place_limit_buy(symbol, price, amount, strategy, meta=meta)
        else:
            internal_id = self.place_limit_sell(symbol, price, amount, strategy, meta=meta)
        
        if internal_id:
            order = self.orders[internal_id]
            return {
                'id': order.order_id,  # Return exchange ID for backward compat
                'internal_id': internal_id,
                'status': order.status,
                'side': side,
                'price': price,
                'qty': amount,
                'info': {},
            }
        raise RuntimeError(f"Failed to place {side} limit order for {symbol}")
    
    def create_market_order(self, symbol: str, side: str, amount: float,
                            strategy: str = 'GRID', meta: dict = None) -> dict:
        """
        Drop-in replacement for exchange.create_market_order().
        """
        if side == 'buy':
            internal_id = self.place_market_buy(symbol, amount, strategy, meta=meta)
        else:
            internal_id = self.place_market_sell(symbol, amount, strategy, meta=meta)
        
        if internal_id:
            order = self.orders[internal_id]
            return {
                'id': order.order_id,
                'internal_id': internal_id,
                'status': order.status,
                'average': order.fill_price,
                'filled': order.qty if order.status == 'filled' else 0,
                'side': side,
                'qty': amount,
                'info': {},
            }
        raise RuntimeError(f"Failed to place {side} market order for {symbol}")
    



# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("OrderManager — Test")
    print("=" * 50)
    
    om = OrderManager()
    
    print(f"\nStats: {om.get_stats()}")
    print(f"P&L: ${om.get_pnl():,.2f}")
    print(f"Trades: {om.get_trade_count()}")
    print(f"\nNo exchanges registered — add one with om.add_exchange('alpaca', adapter)")
    print("Then grid_strategy can use om instead of exchange directly.")
    print("\nTo migrate grid_strategy.py:")
    print("  OLD: self.exchange.create_limit_order(symbol, side, amount, price)")
    print("  NEW: self.order_manager.create_limit_order(symbol, side, amount, price)")
    print("  That's it — same interface, but now centralized.")
