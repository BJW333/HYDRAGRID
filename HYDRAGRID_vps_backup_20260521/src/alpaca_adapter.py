"""
Alpaca Exchange Adapter
========================
Drop-in replacement for ccxt that uses Alpaca's API.
Implements the same interface used by DataFetcher and LiveTrader
so the rest of the bot doesn't need to change.

Supports both crypto and stocks via Alpaca paper/live trading.

Usage:
    from src.alpaca_adapter import AlpacaExchange

    exchange = AlpacaExchange({
        'apiKey': 'YOUR_KEY',
        'secret': 'YOUR_SECRET',
        'paper': True,  # paper trading
    })

    # Same interface as ccxt
    candles = exchange.fetch_ohlcv('BTC/USD', '4h', limit=100)
    balance = exchange.fetch_balance()
    order = exchange.create_market_order('BTC/USD', 'buy', 0.01)
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd


class AlpacaExchange:
    """
    Wraps Alpaca REST API to match the ccxt interface used by the bot.

    ccxt methods implemented:
        fetch_ohlcv(symbol, timeframe, since, limit)
        fetch_balance()
        fetch_ticker(symbol)
        create_market_order(symbol, side, amount)
        load_markets()
        symbols  (property-like)
    """

    # Map bot timeframes → Alpaca bar timeframes
    TIMEFRAME_MAP = {
        '1m':  ('1Min',  1),
        '5m':  ('5Min',  5),
        '15m': ('15Min', 15),
        '30m': ('30Min', 30),
        '1h':  ('1Hour', 60),
        '4h':  ('4Hour', 240),
        '1d':  ('1Day',  1440),
    }

    def __init__(self, config: dict):
        self.api_key = config.get('apiKey', '')
        self.api_secret = config.get('secret', '')
        self.paper = config.get('paper', True)

        # Endpoints
        if self.paper:
            self.trade_url = 'https://paper-api.alpaca.markets'
        else:
            self.trade_url = 'https://api.alpaca.markets'

        self.data_url = 'https://data.alpaca.markets'

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json',
        }

        # Rate limit in ms (mimics ccxt)
        self.rateLimit = 334  # ~3 req/sec

        self._symbols = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _get(self, base_url: str, path: str, params: dict = None) -> dict:
        """GET request with error handling."""
        resp = requests.get(
            f"{base_url}{path}",
            headers=self.headers,
            params=params or {},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, base_url: str, path: str, payload: dict = None) -> dict:
        """POST request with error handling."""
        import json
        resp = requests.post(
            f"{base_url}{path}",
            headers=self.headers,
            data=json.dumps(payload or {}),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _delete(self, base_url: str, path: str) -> dict:
        resp = requests.delete(
            f"{base_url}{path}",
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        if resp.text:
            return resp.json()
        return {}

    @staticmethod
    def _symbol_to_alpaca(symbol: str) -> str:
        """
        Convert ccxt-style symbol to Alpaca format.
        'BTC/USD' → 'BTC/USD' (crypto stays the same on Alpaca)
        'AAPL'    → 'AAPL'    (stocks stay the same)
        """
        return symbol

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        """Detect if symbol is crypto (contains /USD or /USDT)."""
        return '/' in symbol

    # -----------------------------------------------------------------
    # ccxt-compatible: fetch_ohlcv
    # -----------------------------------------------------------------

    def fetch_ohlcv(self, symbol: str, timeframe: str = '4h',
                    since: int = None, limit: int = 100) -> list:
        """
        Fetch OHLCV bars. Returns list of [timestamp_ms, o, h, l, c, v].
        Same format as ccxt.
        """
        tf_info = self.TIMEFRAME_MAP.get(timeframe)
        if not tf_info:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        alpaca_tf, tf_minutes = tf_info

        params = {
            'timeframe': alpaca_tf,
            'limit': min(limit, 10000),
        }

        # Calculate start time
        if since:
            start_dt = datetime.utcfromtimestamp(since / 1000)
        else:
            start_dt = datetime.utcnow() - timedelta(minutes=tf_minutes * limit)

        params['start'] = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        is_crypto = self._is_crypto(symbol)

        if is_crypto:
            # Alpaca crypto data endpoint
            # Symbol format: "BTC/USD" → URL path uses "BTC/USD" encoded
            alpaca_sym = symbol.replace('/', '%2F')
            path = f"/v1beta3/crypto/us/bars"
            params['symbols'] = symbol
        else:
            # Alpaca stock data endpoint
            path = f"/v2/stocks/{symbol}/bars"

        data = self._get(self.data_url, path, params)

        # Parse response into ccxt format: [timestamp_ms, o, h, l, c, v]
        candles = []

        if is_crypto:
            # Crypto response: {"bars": {"BTC/USD": [...]}}
            bars = data.get('bars', {}).get(symbol, [])
        else:
            # Stock response: {"bars": [...]}
            bars = data.get('bars', [])

        for bar in bars:
            ts = bar['t']  # ISO timestamp string
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            ts_ms = int(dt.timestamp() * 1000)

            candles.append([
                ts_ms,
                float(bar['o']),
                float(bar['h']),
                float(bar['l']),
                float(bar['c']),
                float(bar['v']),
            ])

        return candles

    # -----------------------------------------------------------------
    # ccxt-compatible: fetch_balance
    # -----------------------------------------------------------------

    def fetch_balance(self) -> dict:
        """
        Fetch account balance. Returns dict similar to ccxt format.
        """
        account = self._get(self.trade_url, '/v2/account')

        cash = float(account.get('cash', 0))
        equity = float(account.get('equity', 0))
        buying_power = float(account.get('buying_power', 0))

        return {
            'USD': {
                'free': cash,
                'used': equity - cash,
                'total': equity,
            },
            'total': {'USD': equity},
            'free': {'USD': cash},
            'info': account,
        }

    # -----------------------------------------------------------------
    # ccxt-compatible: fetch_ticker
    # -----------------------------------------------------------------

    def fetch_ticker(self, symbol: str) -> dict:
        """
        Fetch current ticker. Returns dict with 'last' price.
        """
        is_crypto = self._is_crypto(symbol)

        if is_crypto:
            path = f"/v1beta3/crypto/us/latest/trades"
            params = {'symbols': symbol}
            data = self._get(self.data_url, path, params)
            trades = data.get('trades', {}).get(symbol, {})
            last_price = float(trades.get('p', 0))
        else:
            path = f"/v2/stocks/{symbol}/trades/latest"
            data = self._get(self.data_url, path)
            last_price = float(data.get('trade', {}).get('p', 0))

        return {
            'symbol': symbol,
            'last': last_price,
            'info': data,
        }

    # -----------------------------------------------------------------
    # ccxt-compatible: create_market_order
    # -----------------------------------------------------------------

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """
        Place a market order.

        Args:
            symbol: e.g. 'BTC/USD' or 'AAPL'
            side: 'buy' or 'sell'
            amount: quantity (fractional ok for crypto)

        Returns:
            dict with 'id', 'average', 'filled', 'status'
        """
        is_crypto = self._is_crypto(symbol)

        payload = {
            'symbol': symbol.replace('/', ''),  # Alpaca order API: 'BTCUSD'
            'side': side,
            'type': 'market',
            'time_in_force': 'gtc' if is_crypto else 'day',
        }

        if is_crypto:
            # Crypto supports notional or qty
            payload['qty'] = str(amount)
        else:
            # Stocks: fractional shares ok
            payload['qty'] = str(amount)

        order = self._post(self.trade_url, '/v2/orders', payload)

        # Wait briefly for fill on paper
        filled_price = None
        for _ in range(10):
            time.sleep(0.5)
            status = self._get(self.trade_url, f"/v2/orders/{order['id']}")
            if status.get('status') in ('filled', 'partially_filled'):
                filled_price = float(status.get('filled_avg_price', 0))
                break

        return {
            'id': order.get('id'),
            'average': filled_price or 0,
            'filled': float(order.get('filled_qty', 0)),
            'status': order.get('status'),
            'info': order,
        }

    # -----------------------------------------------------------------
    # ccxt-compatible: load_markets / symbols
    # -----------------------------------------------------------------

    def load_markets(self):
        """Load available assets."""
        assets = self._get(self.trade_url, '/v2/assets', {'status': 'active'})
        self._symbols = [a['symbol'] for a in assets if a.get('tradable')]

    @property
    def symbols(self) -> list:
        if self._symbols is None:
            self.load_markets()
        return self._symbols

    # -----------------------------------------------------------------
    # Extra Alpaca-specific helpers
    # -----------------------------------------------------------------

    def get_positions(self) -> list:
        """Get all open positions."""
        return self._get(self.trade_url, '/v2/positions')

    def close_all_positions(self):
        """Close all open positions (panic button)."""
        return self._delete(self.trade_url, '/v2/positions')

    def get_account(self) -> dict:
        """Get full account info."""
        return self._get(self.trade_url, '/v2/account')

    # -----------------------------------------------------------------
    # Limit orders (needed for grid trading)
    # -----------------------------------------------------------------

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> dict:
        """Place a limit order."""
        is_crypto = self._is_crypto(symbol)
        payload = {
            'symbol': symbol.replace('/', ''),
            'side': side,
            'type': 'limit',
            'time_in_force': 'gtc',
            'qty': str(amount),
            'limit_price': str(round(price, 2)),
        }
        order = self._post(self.trade_url, '/v2/orders', payload)
        return {
            'id': order.get('id'),
            'status': order.get('status'),
            'side': side,
            'price': price,
            'qty': amount,
            'info': order,
        }

    def get_order(self, order_id: str) -> dict:
        """Check status of a specific order."""
        data = self._get(self.trade_url, f'/v2/orders/{order_id}')
        return {
            'id': data.get('id'),
            'status': data.get('status'),
            'filled_qty': float(data.get('filled_qty', 0)),
            'filled_avg_price': float(data.get('filled_avg_price', 0)) if data.get('filled_avg_price') else 0,
            'side': data.get('side'),
            'info': data,
        }

    def get_open_orders(self, symbol: str = None) -> list:
        """Get all open orders, optionally filtered by symbol."""
        params = {'status': 'open'}
        if symbol:
            params['symbols'] = symbol.replace('/', '')
        return self._get(self.trade_url, '/v2/orders', params)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a specific order."""
        return self._delete(self.trade_url, f'/v2/orders/{order_id}')

    def cancel_all_orders(self) -> dict:
        """Cancel all open orders."""
        return self._delete(self.trade_url, '/v2/orders')

