"""
Indicators Module
=================
Technical indicator calculations for the breakout strategy.
All functions are pure - they take data in and return calculated values.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    BB_LENGTH, BB_STD, ATR_PERIOD, TREND_MA_LENGTH,
    VOLUME_MA_LENGTH, LOOKBACK_PERIOD, TREND_MA_SLOPE_LOOKBACK
)


def calculate_bollinger_bands(close: pd.Series, length: int = BB_LENGTH,
                              std_dev: float = BB_STD) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands and Band Width.
    
    Args:
        close: Series of closing prices
        length: Period for moving average
        std_dev: Number of standard deviations
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band, bb_width)
    """
    # Middle band is SMA
    middle_band = close.rolling(window=length).mean()
    
    # Standard deviation
    rolling_std = close.rolling(window=length).std()
    
    # Upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    # Band width (normalized by middle band for comparison across price levels)
    bb_width = (upper_band - lower_band) / middle_band
    
    return upper_band, middle_band, lower_band, bb_width


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = ATR_PERIOD) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    ATR measures market volatility by decomposing the entire range of an asset
    price for that period.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR period
        
    Returns:
        ATR series
    """
    # True Range is the maximum of:
    # 1. Current high - current low
    # 2. Abs(current high - previous close)
    # 3. Abs(current low - previous close)
    
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the smoothed average of True Range
    # Using Wilder's smoothing method (similar to EMA)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr


def calculate_moving_average(data: pd.Series, period: int,
                             ma_type: str = 'sma') -> pd.Series:
    """
    Calculate moving average.
    
    Args:
        data: Series of values
        period: MA period
        ma_type: 'sma' for Simple, 'ema' for Exponential
        
    Returns:
        Moving average series
    """
    if ma_type.lower() == 'sma':
        return data.rolling(window=period).mean()
    elif ma_type.lower() == 'ema':
        return data.ewm(span=period, adjust=False).mean()
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")


def calculate_volume_ma(volume: pd.Series,
                        period: int = VOLUME_MA_LENGTH) -> pd.Series:
    """
    Calculate volume moving average.
    
    Args:
        volume: Series of volume data
        period: MA period
        
    Returns:
        Volume MA series
    """
    return volume.rolling(window=period).mean()


def calculate_price_range(high: pd.Series, low: pd.Series,
                          period: int = LOOKBACK_PERIOD) -> pd.Series:
    """
    Calculate price range as a percentage over the lookback period.
    
    This measures how "tight" the consolidation is.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        period: Lookback period
        
    Returns:
        Series of range percentages
    """
    rolling_high = high.rolling(window=period).max()
    rolling_low = low.rolling(window=period).min()
    
    # Range as percentage of the midpoint
    midpoint = (rolling_high + rolling_low) / 2
    range_pct = (rolling_high - rolling_low) / midpoint
    
    return range_pct


def calculate_ma_slope(ma_series: pd.Series,
                       lookback: int = TREND_MA_SLOPE_LOOKBACK) -> pd.Series:
    """
    Calculate the slope of a moving average.
    
    Positive slope = uptrend
    Negative slope = downtrend
    
    Args:
        ma_series: Moving average series
        lookback: Number of periods to measure slope
        
    Returns:
        Slope series (price change over lookback period)
    """
    return ma_series.diff(lookback)


def calculate_bb_width_percentile(bb_width: pd.Series,
                                  period: int = LOOKBACK_PERIOD) -> pd.Series:
    """
    Calculate where current BB width sits relative to recent history.
    
    Low percentile = squeeze (bands tight)
    High percentile = expansion (bands wide)
    
    Args:
        bb_width: Bollinger Band width series
        period: Lookback period for percentile calculation
        
    Returns:
        Percentile series (0-100)
    """
    def rolling_percentile(x):
        if len(x) < period:
            return np.nan
        return (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50
    
    return bb_width.rolling(window=period).apply(rolling_percentile, raw=False)


def is_bb_squeeze(bb_width: pd.Series, period: int = LOOKBACK_PERIOD) -> pd.Series:
    """
    Detect if Bollinger Bands are in a squeeze (width at period low).
    
    Args:
        bb_width: Bollinger Band width series
        period: Lookback period
        
    Returns:
        Boolean series (True = squeeze active)
    """
    rolling_min = bb_width.rolling(window=period).min()
    
    # Squeeze is when current width equals the rolling minimum
    # Using small tolerance for float comparison
    return (bb_width - rolling_min).abs() < (bb_width * 0.001)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Optional indicator for additional confirmation.
    
    Args:
        close: Series of closing prices
        period: RSI period
        
    Returns:
        RSI series (0-100)
    """
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  #Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all required indicators to a DataFrame.
    
    This is the main function to prepare data for signal generation.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
        
    Returns:
        DataFrame with all indicators added
    """
    df = df.copy()
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'] = \
        calculate_bollinger_bands(df['close'])
    
    # BB Squeeze detection
    df['bb_squeeze'] = is_bb_squeeze(df['bb_width'])
    df['bb_width_percentile'] = calculate_bb_width_percentile(df['bb_width'])
    
    # ATR
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    
    # Trend MA
    df['ma_50'] = calculate_moving_average(df['close'], TREND_MA_LENGTH)
    df['ma_slope'] = calculate_ma_slope(df['ma_50'])
    
    # Volume MA
    df['volume_ma'] = calculate_volume_ma(df['volume'])
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Price range
    df['price_range'] = calculate_price_range(df['high'], df['low'])
    
    # Consolidation range boundaries
    df['range_high'] = df['high'].rolling(window=LOOKBACK_PERIOD).max()
    df['range_low'] = df['low'].rolling(window=LOOKBACK_PERIOD).min()
    
    # RSI (optional)
    df['rsi'] = calculate_rsi(df['close'])
    
    return df


def get_indicator_summary(df: pd.DataFrame) -> dict:
    """
    Get summary of current indicator values.
    
    Useful for live trading to quickly assess current state.
    
    Args:
        df: DataFrame with indicators
        
    Returns:
        Dictionary of current indicator values
    """
    latest = df.iloc[-1]
    
    return {
        'price': latest['close'],
        'bb_upper': latest['bb_upper'],
        'bb_lower': latest['bb_lower'],
        'bb_width': latest['bb_width'],
        'bb_squeeze': latest['bb_squeeze'],
        'atr': latest['atr'],
        'ma_50': latest['ma_50'],
        'ma_slope': latest['ma_slope'],
        'volume_ratio': latest['volume_ratio'],
        'price_range': latest['price_range'],
        'range_high': latest['range_high'],
        'range_low': latest['range_low'],
        'rsi': latest['rsi'],
        'trend': 'UP' if latest['ma_slope'] > 0 else 'DOWN',
        'price_vs_ma': 'ABOVE' if latest['close'] > latest['ma_50'] else 'BELOW'
    }


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    # Create sample data for testing
    print("Testing Indicators Module...")
    print("=" * 50)
    
    # Generate synthetic price data
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range(start='2024-01-01', periods=n, freq='4h')
    
    # Random walk for close prices
    close = 3000 + np.cumsum(np.random.randn(n) * 20)
    
    # Generate OHLCV
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 5,
        'high': close + abs(np.random.randn(n) * 15),
        'low': close - abs(np.random.randn(n) * 15),
        'close': close,
        'volume': np.random.randint(1000, 10000, n).astype(float)
    }, index=dates)
    
    # Add indicators
    df = add_all_indicators(df)
    
    print("\nDataFrame with indicators:")
    print(df.tail(10).to_string())
    
    print("\n\nCurrent Indicator Summary:")
    summary = get_indicator_summary(df)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
