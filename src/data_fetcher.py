"""
Data Fetcher Module
===================
Handles all data retrieval from exchanges and local storage.
"""

try:
    import ccxt
except ImportError:
    ccxt = None  # Not needed when using Alpaca
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import EXCHANGE, SYMBOL, TIMEFRAME, API_KEY, API_SECRET


class DataFetcher:
    """
    Fetches OHLCV data from cryptocurrency exchanges.
    
    Supports:
    - Alpaca (paper + live trading)
    - Any ccxt-compatible exchange (Kraken, etc.)
    - Historical data retrieval
    - Live data streaming
    - Local caching to avoid redundant API calls
    """
    
    def __init__(self, exchange_id: str = EXCHANGE, api_key: str = API_KEY, 
                 api_secret: str = API_SECRET):
        self.exchange_id = exchange_id
        self.exchange = self._connect_exchange(exchange_id, api_key, api_secret)
        self.data_dir = Path(__file__).parent.parent / "data" / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _connect_exchange(self, exchange_id: str, api_key: str, 
                          api_secret: str):
        """
        Create exchange connection. Uses AlpacaExchange for 'alpaca',
        otherwise falls back to ccxt.
        """
        if exchange_id == 'alpaca':
            from src.alpaca_adapter import AlpacaExchange
            from config.settings import PAPER_TRADING
            return AlpacaExchange({
                'apiKey': api_key,
                'secret': api_secret,
                'paper': PAPER_TRADING,
            })
        
        # ccxt fallback
        exchange_class = getattr(ccxt, exchange_id)
        config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        }
        if api_key and api_secret:
            config['apiKey'] = api_key
            config['secret'] = api_secret
        return exchange_class(config)
    
    def fetch_historical(self, symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                         start_date: str = None, end_date: str = None,
                         limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading pair (e.g., 'ETH/USDT')
            timeframe: Candle timeframe (e.g., '4h', '1d')
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            limit: Max candles per request
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Convert dates to timestamps
        if start_date:
            since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            # Default to 6 months ago
            since = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
        
        if end_date:
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)
        
        all_candles = []
        current_since = since
        
        print(f"Fetching {symbol} {timeframe} data from {start_date or '6 months ago'}...")
        
        while current_since < end_ts:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Move to next batch
                current_since = candles[-1][0] + 1
                
                # Progress indicator
                print(f"  Fetched {len(all_candles)} candles...", end='\r')
                
                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"\nError fetching data: {e}")
                break
        
        print(f"\nTotal candles fetched: {len(all_candles)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by time
        df.sort_index(inplace=True)
        
        return df
    
    def fetch_live(self, symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                   num_candles: int = 100) -> pd.DataFrame:
        """
        Fetch recent candles for live trading.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            num_candles: Number of recent candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            candles = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=num_candles
            )
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str = SYMBOL) -> float:
        """
        Get current market price.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Current price as float
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching price: {e}")
            return None
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = None,
                    symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> str:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Custom filename (optional)
            symbol: Symbol for auto-naming
            timeframe: Timeframe for auto-naming
            
        Returns:
            Path to saved file
        """
        if filename is None:
            # Auto-generate filename
            symbol_clean = symbol.replace('/', '_')
            filename = f"{symbol_clean}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        filepath = self.data_dir / filename
        df.to_csv(filepath)
        print(f"Data saved to: {filepath}")
        
        return str(filepath)
    
    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """
        Load DataFrame from CSV file.
        
        Args:
            filename: Name of file in data directory
            
        Returns:
            DataFrame with OHLCV data
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        
        print(f"Loaded {len(df)} candles from {filepath}")
        
        return df
    
    def list_cached_data(self) -> list:
        """
        List all cached data files.
        
        Returns:
            List of filenames
        """
        files = list(self.data_dir.glob("*.csv"))
        return [f.name for f in files]
    
    def get_available_symbols(self) -> list:
        """
        Get list of tradeable symbols on the exchange.
        
        Returns:
            List of symbol strings
        """
        self.exchange.load_markets()
        return list(self.exchange.symbols)


# =============================================================================
# Convenience functions for quick access
# =============================================================================

def fetch_eth_data(timeframe: str = '4h', days: int = 180) -> pd.DataFrame:
    """
    Quick function to fetch ETH/USDT data.
    
    Args:
        timeframe: Candle timeframe
        days: Number of days of history
        
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = DataFetcher()
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    return fetcher.fetch_historical(
        symbol='ETH/USDT',
        timeframe=timeframe,
        start_date=start_date
    )


def fetch_btc_data(timeframe: str = '4h', days: int = 180) -> pd.DataFrame:
    """
    Quick function to fetch BTC/USDT data.
    
    Args:
        timeframe: Candle timeframe
        days: Number of days of history
        
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = DataFetcher()
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    return fetcher.fetch_historical(
        symbol='BTC/USDT',
        timeframe=timeframe,
        start_date=start_date
    )


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Data Fetcher...")
    print("=" * 50)
    
    fetcher = DataFetcher()
    
    # Fetch sample data
    df = fetcher.fetch_historical(
        symbol='ETH/USDT',
        timeframe='4h',
        start_date='2024-06-01',
        end_date='2024-12-31'
    )
    
    print("\nData Preview:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Save to CSV
    fetcher.save_to_csv(df)
