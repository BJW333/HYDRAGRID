"""
HYDRA-SPOT Strategy
===================
Regime-gated multi-engine strategy adapted for Kraken SPOT trading.

Based on HYDRA-X concepts but:
- No ML required (rule-based regime detection + quality filtering)
- Optimized for Kraken's fee structure (0.16-0.26%)
- Spot-only indicators (no funding rate / open interest)

Key innovations from HYDRA-X:
1. REGIME DETECTION - Only trade strategies that match current market state
2. MULTIPLE ENGINES - Different alpha sources for different conditions
3. QUALITY GATING - Only take high-probability setups
4. SMART EXITS - TP1 → move stop to breakeven → TP2
5. ADAPTIVE SIZING - Bigger on better setups, smaller when struggling

Target: 20-40 high-quality trades/week across all assets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config.settings import COMMISSION, SLIPPAGE

class Regime(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    MEAN_REVERT = "mean_revert"
    COMPRESSION = "compression"
    VOLATILE = "volatile"
    CHOPPY = "choppy"


@dataclass
class CandidateTrade:
    """A potential trade from an engine."""
    symbol: str
    timestamp: pd.Timestamp
    direction: int  # +1 long, -1 short
    engine: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    ttl_bars: int  # Time-to-live in bars (fast fail if no movement)
    quality_score: float  # 0-1 score from quality filter
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class HydraConfig:
    """Strategy configuration - tuned for Kraken spot."""
    
    # === REGIME THRESHOLDS ===
    trend_eff_ratio_threshold: float = 0.50  # Above = trending
    compression_tr_z_threshold: float = -0.8  # Below = compressed
    volatile_tr_z_threshold: float = 1.2  # Above = volatile
    
    # === QUALITY GATING (KEY FOR PROFITABILITY) ===
    #min_quality_score: float = 0.48  # Slightly lower
    min_quality_score: float = 0.85 # Data shows quality >= 0.85 filters most false signals
    #min_regime_confidence: float = 0.35  # Allow more regime situations
    min_regime_confidence: float = 0.65 # Only take trades when regime detection is strong - critical for spot where we need high winrate
    
    # === RISK MANAGEMENT ===
    base_risk_pct: float = 0.10  # 10% risk per trade — match settings
    max_positions: int = 3  # Per symbol
    max_total_positions: int = 12  # Across all symbols
    daily_loss_limit_pct: float = 0.15  # 15% daily loss limit
     
    # === DRAWDOWN BRAKES ===
    drawdown_brake_start: float = 0.05  # Start reducing size at 5% DD
    drawdown_brake_max: float = 0.15  # Max reduction at 15% DD
    
    # === TRUST BRAKE (adapts to recent performance) ===
    #trust_window_trades: int = 30
    trust_window_trades: int = 20 # Shorter window for more responsiveness in spot trading
    #trust_min_winrate: float = 0.38  # Below this = reduce size
    trust_min_winrate: float = 0.45 # Only reduce size if recent winrate is below 45% - we want to be aggressive in spot but still protect capital if we're in a rough patch
    
    # === COSTS (Kraken spot) ===
    fee_pct: float = COMMISSION  # Use same fee as backtester (0.26% taker)
    slippage_pct: float = SLIPPAGE  # Use same slippage as backtester
    
    # === ENGINE PARAMETERS (tuned for 15m timeframe) ===
    # Volatility Snapback
    #snapback_vwap_threshold: float = 2.0 # How far from VWAP to trigger snapback (in RV-adjusted std devs)
    snapback_vwap_threshold: float = 2.8 # More aggressive snapback threshold for spot - we want to catch bigger moves
    #snapback_tr_z_min: float = 1.0
    snapback_tr_z_min: float = 1.5
    #snapback_ttl: int = 60  # 60 bars = 15 hours on 15m
    snapback_ttl: int = 36  # 36 bars = 9 hours — trades need time to play out
    
    # Compression Breakout
    #compress_ttl: int = 72  # 72 bars = 18 hours on 15m
    compress_ttl: int = 48 # 48 bars = 12 hours — 34 of 52 trades expired on TTL, need more room
    
    # Trend Pullback
    #pullback_vwap_range: Tuple[float, float] = (0.3, 1.2)
    pullback_vwap_range: Tuple[float, float] = (0.55, 0.95) # Tighter pullback range for spot - we want to catch more precise pullbacks in a strong trend
    #pullback_ttl: int = 80  # 80 bars = 20 hours on 15m
    pullback_ttl: int = 40  # Shorter TTL for pullback - we want to quickly fail if it doesn't move in our favor, since spot can be more volatile and we want to protect capital
    
    # Momentum Continuation
    #momentum_min_ret: float = 0.003
    momentum_min_ret: float = 0.007
    #momentum_ttl: int = 72  # 72 bars = 18 hours
    momentum_ttl: int = 48 # 48 bars = 12 hours — momentum_continuation is the profitable engine, give it room


class FeatureEngine:
    """Calculate all features needed for regime detection and signal generation."""
    
    def __init__(self, vol_lookback: int = 60):
        self.vol_lookback = vol_lookback
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all features to dataframe."""
        df = df.copy()
        
        # === RETURNS ===
        df['ret1'] = df['close'].pct_change()
        df['ret5'] = df['close'].pct_change(5)
        df['ret15'] = df['close'].pct_change(15)
        df['ret_log1'] = np.log(df['close']).diff()
        
        # === VOLATILITY ===
        df['rv'] = df['ret_log1'].rolling(self.vol_lookback).std() * np.sqrt(self.vol_lookback)
        df['rv'] = df['rv'].replace(0, np.nan).ffill().fillna(0.01)
        
        # True Range z-score
        df['tr'] = (df['high'] - df['low']) / df['close'].replace(0, 1)
        tr_mean = df['tr'].rolling(100).mean()
        tr_std = df['tr'].rolling(100).std()
        df['tr_z'] = (df['tr'] - tr_mean) / (tr_std + 1e-9)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr_raw = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr_raw.rolling(14).mean()
        
        # === TREND ===
        abs_move = df['close'].diff(20).abs()
        sum_abs_moves = df['close'].diff().abs().rolling(20).sum()
        df['eff_ratio'] = abs_move / (sum_abs_moves + 1e-9)
        
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=50, adjust=False).mean()
        
        df['trend_dir'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['above_trend_ema'] = (df['close'] > df['ema_trend']).astype(int)
        
        # === VWAP ===
        typical = (df['high'] + df['low'] + df['close']) / 3
        vol_sum = df['volume'].rolling(50).sum()
        vwap_sum = (typical * df['volume']).rolling(50).sum()
        df['vwap'] = vwap_sum / (vol_sum + 1e-9)
        
        df['vwap_dist'] = (df['close'] - df['vwap']) / (df['rv'] * df['close'] + 1e-9)
        
        # === VOLUME ===
        df['vol_ma'] = df['volume'].rolling(50).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1e-9)
        
        # === RSI ===
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # === MOMENTUM ===
        df['momentum'] = df['close'].pct_change(5)
        df['momentum_accel'] = df['momentum'].diff(3)
        
        # === CANDLE ANALYSIS ===
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['close']
        df['body_ratio'] = abs(df['body']) / (df['high'] - df['low'] + 1e-9)
        
        df['strong_bullish'] = (df['body'] > 0) & (df['body_ratio'] > 0.5)  # Lowered from 0.6
        df['strong_bearish'] = (df['body'] < 0) & (df['body_ratio'] > 0.5)  # Lowered from 0.6
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df


class RegimeDetector:
    """Rule-based regime classification."""
    
    def __init__(self, cfg: HydraConfig):
        self.cfg = cfg
    
    def detect(self, feats: pd.DataFrame) -> Dict[str, float]:
        """Detect current market regime."""
        if len(feats) < 50:
            return {"choppy": 1.0}
        
        r = feats.iloc[-1]
        regimes = {}
        
        eff = r['eff_ratio'] if not pd.isna(r['eff_ratio']) else 0.3
        trend_dir = r['trend_dir'] if not pd.isna(r['trend_dir']) else 0
        above_ema = r['above_trend_ema'] if not pd.isna(r['above_trend_ema']) else 0.5
        tr_z = r['tr_z'] if not pd.isna(r['tr_z']) else 0
        rsi = r['rsi'] if not pd.isna(r['rsi']) else 50
        
        # TREND
        if eff > self.cfg.trend_eff_ratio_threshold:
            if trend_dir > 0 and above_ema:
                regimes['trend_up'] = float(np.clip(eff + 0.2, 0, 1))
            elif trend_dir < 0 and not above_ema:
                regimes['trend_down'] = float(np.clip(eff + 0.2, 0, 1))
        
        # COMPRESSION
        if tr_z < self.cfg.compression_tr_z_threshold:
            regimes['compression'] = float(np.clip(abs(tr_z) / 2, 0, 1))
        
        # VOLATILE
        if tr_z > self.cfg.volatile_tr_z_threshold:
            regimes['volatile'] = float(np.clip(tr_z / 2, 0, 1))
        
        # MEAN REVERT
        if eff < 0.35:
            if rsi < 30 or rsi > 70:
                regimes['mean_revert'] = float(np.clip(0.5 + abs(50 - rsi) / 100, 0, 1))
        
        if not regimes or max(regimes.values()) < 0.4:
            regimes['choppy'] = 0.6
        
        total = sum(regimes.values())
        if total > 0:
            regimes = {k: v / total for k, v in regimes.items()}
        
        return regimes


class AlphaEngines:
    """Multiple alpha engines that generate trade candidates."""
    
    def __init__(self, cfg: HydraConfig):
        self.cfg = cfg
        
        self.engine_regimes = {
            'volatility_snapback': ['volatile'],
            'compression_breakout': ['compression'],
            'trend_pullback': ['trend_up', 'trend_down'],
            'momentum_continuation': ['trend_up', 'trend_down'],
            'vwap_reversion': ['mean_revert'],
        }
    
    def generate(self, symbol: str, df: pd.DataFrame, feats: pd.DataFrame,
                 regime: Dict[str, float]) -> List[CandidateTrade]:
        """Generate all candidate trades from all engines."""
        candidates = []
        
        if len(feats) < 50:
            return candidates
        
        r = feats.iloc[-1]
        close = float(df.iloc[-1]['close'])
        ts = df.index[-1]
        
        rv = float(r['rv']) if not pd.isna(r['rv']) else 0.01
        atr = float(r['atr']) if not pd.isna(r['atr']) else close * 0.01
        base_stop = max(atr * 1.5, close * rv * 1.5, close * 0.006)
        
        # ENGINE 1: VOLATILITY SNAPBACK
        if self._regime_score(regime, 'volatility_snapback') > 0.3:
            vwap_dist = float(r['vwap_dist']) if not pd.isna(r['vwap_dist']) else 0
            tr_z = float(r['tr_z']) if not pd.isna(r['tr_z']) else 0
            ret1 = float(r['ret1']) if not pd.isna(r['ret1']) else 0
            
            # Long snapback - price extended below VWAP
            #if tr_z > 0.8 and vwap_dist < -1.5:
            if tr_z > self.cfg.snapback_tr_z_min and vwap_dist < -self.cfg.snapback_vwap_threshold:
                if ret1 < 0:
                    quality = self._calc_quality(r, 'snapback_long', regime)
                    candidates.append(CandidateTrade(
                        symbol=symbol, timestamp=ts, direction=1, engine='volatility_snapback',
                        entry=close, stop=close - base_stop * 1.2,
                        tp1=close + base_stop * 1.0, tp2=close + base_stop * 2.5,
                        ttl_bars=self.cfg.snapback_ttl, quality_score=quality,
                        meta={'vwap_dist': vwap_dist, 'tr_z': tr_z}
                    ))
            
            # Short snapback - price extended above VWAP
            #if tr_z > 0.8 and vwap_dist > 1.5:
            if tr_z > self.cfg.snapback_tr_z_min and vwap_dist > self.cfg.snapback_vwap_threshold:
                if ret1 > 0:
                    quality = self._calc_quality(r, 'snapback_short', regime)
                    candidates.append(CandidateTrade(
                        symbol=symbol, timestamp=ts, direction=-1, engine='volatility_snapback',
                        entry=close, stop=close + base_stop * 1.2,
                        tp1=close - base_stop * 1.0, tp2=close - base_stop * 2.5,
                        ttl_bars=self.cfg.snapback_ttl, quality_score=quality,
                        meta={'vwap_dist': vwap_dist, 'tr_z': tr_z}
                    ))
        
        # ENGINE 2: COMPRESSION BREAKOUT
        if self._regime_score(regime, 'compression_breakout') > 0.3:
            tr_z = float(r['tr_z']) if not pd.isna(r['tr_z']) else 0
            eff = float(r['eff_ratio']) if not pd.isna(r['eff_ratio']) else 0.5
            ret5 = float(r['ret5']) if not pd.isna(r['ret5']) else 0
            vol_ratio = float(r['vol_ratio']) if not pd.isna(r['vol_ratio']) else 1
            
            # Compression detected
            if tr_z < -0.5 and eff < 0.45:
                vol_confirm = vol_ratio > 1.0  # Lowered from 1.2
                
                if ret5 > 0.001 and r['strong_bullish'] and vol_confirm:
                    quality = self._calc_quality(r, 'breakout_long', regime)
                    candidates.append(CandidateTrade(
                        symbol=symbol, timestamp=ts, direction=1, engine='compression_breakout',
                        entry=close, stop=close - base_stop,
                        tp1=close + base_stop * 1.2, tp2=close + base_stop * 3.0,
                        ttl_bars=self.cfg.compress_ttl, quality_score=quality,
                        meta={'tr_z': tr_z, 'eff_ratio': eff}
                    ))
                
                if ret5 < -0.001 and r['strong_bearish'] and vol_confirm:
                    quality = self._calc_quality(r, 'breakout_short', regime)
                    candidates.append(CandidateTrade(
                        symbol=symbol, timestamp=ts, direction=-1, engine='compression_breakout',
                        entry=close, stop=close + base_stop,
                        tp1=close - base_stop * 1.2, tp2=close - base_stop * 3.0,
                        ttl_bars=self.cfg.compress_ttl, quality_score=quality,
                        meta={'tr_z': tr_z, 'eff_ratio': eff}
                    ))
        
        # ENGINE 3: TREND PULLBACK
        if self._regime_score(regime, 'trend_pullback') > 0.25:
            eff = float(r['eff_ratio']) if not pd.isna(r['eff_ratio']) else 0.5
            vwap_dist = float(r['vwap_dist']) if not pd.isna(r['vwap_dist']) else 0
            rsi = float(r['rsi']) if not pd.isna(r['rsi']) else 50
            trend_dir = float(r['trend_dir']) if not pd.isna(r['trend_dir']) else 0
            
            # Long pullback - in uptrend, price pulled back
            if (regime.get('trend_up', 0) > 0.25 or trend_dir > 0) and eff > 0.35:
                #if -1.8 < vwap_dist < -0.15:  # Widened range
                low_pb, high_pb = self.cfg.pullback_vwap_range
                if -high_pb < vwap_dist < -low_pb:
                    if r['strong_bullish'] and rsi > 32:
                        quality = self._calc_quality(r, 'pullback_long', regime)
                        candidates.append(CandidateTrade(
                            symbol=symbol, timestamp=ts, direction=1, engine='trend_pullback',
                            entry=close, stop=close - base_stop * 1.3,
                            tp1=close + base_stop * 1.5, tp2=close + base_stop * 3.5,
                            ttl_bars=self.cfg.pullback_ttl, quality_score=quality,
                            meta={'eff_ratio': eff, 'vwap_dist': vwap_dist}
                        ))
            
            # Short pullback - in downtrend, price rallied
            if (regime.get('trend_down', 0) > 0.25 or trend_dir < 0) and eff > 0.35:
                #if 0.15 < vwap_dist < 1.8:  # Widened range
                low_pb, high_pb = self.cfg.pullback_vwap_range
                if low_pb < vwap_dist < high_pb:    
                    if r['strong_bearish'] and rsi < 68:
                        quality = self._calc_quality(r, 'pullback_short', regime)
                        candidates.append(CandidateTrade(
                            symbol=symbol, timestamp=ts, direction=-1, engine='trend_pullback',
                            entry=close, stop=close + base_stop * 1.3,
                            tp1=close - base_stop * 1.5, tp2=close - base_stop * 3.5,
                            ttl_bars=self.cfg.pullback_ttl, quality_score=quality,
                            meta={'eff_ratio': eff, 'vwap_dist': vwap_dist}
                        ))
        
        # ENGINE 4: MOMENTUM CONTINUATION
        # Only fire when we have CLEAR trend + momentum + volume
        trend_score = max(regime.get('trend_up', 0), regime.get('trend_down', 0), 0)
        if trend_score > 0.35:  # Slightly lower
            momentum = float(r['momentum']) if not pd.isna(r['momentum']) else 0
            mom_accel = float(r['momentum_accel']) if not pd.isna(r['momentum_accel']) else 0
            vol_ratio = float(r['vol_ratio']) if not pd.isna(r['vol_ratio']) else 1
            rsi = float(r['rsi']) if not pd.isna(r['rsi']) else 50
            eff = float(r['eff_ratio']) if not pd.isna(r['eff_ratio']) else 0.3
            vol_confirm = vol_ratio > 1.2  # Slightly lower
            
            # Long momentum - need efficiency ratio + momentum + volume ALL aligned
            #if momentum > 0.003 and eff > 0.40 and vol_confirm:
            if momentum > self.cfg.momentum_min_ret and eff > 0.40 and vol_confirm:
                if r['strong_bullish'] and 40 < rsi < 75:
                    quality = self._calc_quality(r, 'momentum_long', regime)
                    candidates.append(CandidateTrade(
                        symbol=symbol, timestamp=ts, direction=1, engine='momentum_continuation',
                        entry=close, stop=close - base_stop * 1.2,
                        tp1=close + base_stop * 1.3, tp2=close + base_stop * 2.8,
                        ttl_bars=self.cfg.momentum_ttl, quality_score=quality,
                        meta={'momentum': momentum, 'vol_ratio': vol_ratio, 'eff': eff}
                    ))
            
            # Short momentum
            #if momentum < -0.003 and eff > 0.40 and vol_confirm:
            if momentum < -self.cfg.momentum_min_ret and eff > 0.40 and vol_confirm:
                if r['strong_bearish'] and 25 < rsi < 60:
                    quality = self._calc_quality(r, 'momentum_short', regime)
                    candidates.append(CandidateTrade(
                        symbol=symbol, timestamp=ts, direction=-1, engine='momentum_continuation',
                        entry=close, stop=close + base_stop * 1.2,
                        tp1=close - base_stop * 1.3, tp2=close - base_stop * 2.8,
                        ttl_bars=self.cfg.momentum_ttl, quality_score=quality,
                        meta={'momentum': momentum, 'vol_ratio': vol_ratio, 'eff': eff}
                    ))
        
        # ENGINE 5: VWAP REVERSION
        if self._regime_score(regime, 'vwap_reversion') > 0.25:
            vwap_dist = float(r['vwap_dist']) if not pd.isna(r['vwap_dist']) else 0
            rsi = float(r['rsi']) if not pd.isna(r['rsi']) else 50
            
            # Long reversion - oversold below VWAP
            if vwap_dist < -1.2 and rsi < 40:
                if r['strong_bullish']:
                    quality = self._calc_quality(r, 'reversion_long', regime)
                    candidates.append(CandidateTrade(
                        symbol=symbol, timestamp=ts, direction=1, engine='vwap_reversion',
                        entry=close, stop=close - base_stop,
                        tp1=close + base_stop * 1.0, tp2=close + base_stop * 2.2,
                        ttl_bars=20, quality_score=quality,
                        meta={'vwap_dist': vwap_dist, 'rsi': rsi}
                    ))
            
            # Short reversion - overbought above VWAP
            if vwap_dist > 1.2 and rsi > 60:
                if r['strong_bearish']:
                    quality = self._calc_quality(r, 'reversion_short', regime)
                    candidates.append(CandidateTrade(
                        symbol=symbol, timestamp=ts, direction=-1, engine='vwap_reversion',
                        entry=close, stop=close + base_stop,
                        tp1=close - base_stop * 1.0, tp2=close - base_stop * 2.2,
                        ttl_bars=20, quality_score=quality,
                        meta={'vwap_dist': vwap_dist, 'rsi': rsi}
                    ))
        
        return candidates
    
    def _regime_score(self, regime: Dict[str, float], engine: str) -> float:
        """Get combined regime score for an engine."""
        relevant = self.engine_regimes.get(engine, [])
        return sum(regime.get(r, 0) for r in relevant)
    
    def _calc_quality(self, r: pd.Series, signal_type: str, regime: Dict[str, float]) -> float:
        """Calculate quality score (0-1) for a candidate."""
        score = 0.5
        
        vol_ratio = float(r['vol_ratio']) if not pd.isna(r['vol_ratio']) else 1.0
        if vol_ratio > 1.5:
            score += 0.15
        elif vol_ratio > 1.2:
            score += 0.08
        elif vol_ratio < 0.7:
            score -= 0.10
        
        body_ratio = float(r['body_ratio']) if not pd.isna(r['body_ratio']) else 0.5
        if body_ratio > 0.7:
            score += 0.10
        
        rsi = float(r['rsi']) if not pd.isna(r['rsi']) else 50
        if 'long' in signal_type:
            if rsi < 70:
                score += 0.05
            if rsi < 50:
                score += 0.05
            if rsi > 80:
                score -= 0.15
        else:
            if rsi > 30:
                score += 0.05
            if rsi > 50:
                score += 0.05
            if rsi < 20:
                score -= 0.15
        
        max_regime = max(regime.values()) if regime else 0
        if max_regime > 0.6:
            score += 0.10
        
        eff = float(r['eff_ratio']) if not pd.isna(r['eff_ratio']) else 0.5
        if 'pullback' in signal_type or 'momentum' in signal_type:
            if eff > 0.5:
                score += 0.10
        elif 'reversion' in signal_type or 'snapback' in signal_type:
            if eff < 0.4:
                score += 0.08
        
        return float(np.clip(score, 0, 1))


@dataclass
class Position:
    """Active position tracking."""
    symbol: str
    direction: int
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    size: float
    engine: str
    entry_bar: int
    ttl_bars: int
    tp1_taken: bool = False
    active: bool = True


class HydraSpotStrategy:
    """Main HYDRA-SPOT strategy class."""
    
    def __init__(self, cfg: HydraConfig = None):
        self.cfg = cfg or HydraConfig()
        self.features = FeatureEngine()
        self.regime_detector = RegimeDetector(self.cfg)
        self.engines = AlphaEngines(self.cfg)
        
        self.positions: List[Position] = []
        self.equity = 10000.0
        self.starting_equity = 10000.0
        self.peak_equity = 10000.0
        self.day_start_equity = 10000.0
        
        self.trade_results: List[int] = []
        self.trades_today = 0
        self.last_trade_bar: Dict[str, int] = {}
        self.cooldown_bars = 3  # 3 bars = 45 min on 15m
    
    def on_bar(self, symbol: str, candles: pd.DataFrame, bar_idx: int) -> List[dict]:
        """Process a new bar."""
        actions = []
        
        exit_actions = self._manage_positions(symbol, candles, bar_idx)
        actions.extend(exit_actions)
        
        daily_pnl_pct = (self.equity - self.day_start_equity) / self.day_start_equity
        if daily_pnl_pct <= -self.cfg.daily_loss_limit_pct:
            return actions
        
        symbol_positions = sum(1 for p in self.positions if p.active and p.symbol == symbol)
        total_positions = sum(1 for p in self.positions if p.active)
        
        if symbol_positions >= self.cfg.max_positions:
            return actions
        if total_positions >= self.cfg.max_total_positions:
            return actions
        
        last_bar = self.last_trade_bar.get(symbol, -100)
        if bar_idx - last_bar < self.cooldown_bars:
            return actions
        
        feats = self.features.compute(candles)
        if len(feats) < 100:
            return actions
        
        regime = self.regime_detector.detect(feats)
        max_regime_conf = max(regime.values()) if regime else 0
        
        if max_regime_conf < self.cfg.min_regime_confidence:
            return actions
        
        candidates = self.engines.generate(symbol, candles, feats, regime)
        
        if not candidates:
            return actions
        
        qualified = [c for c in candidates if c.quality_score >= self.cfg.min_quality_score]
        
        if not qualified:
            return actions
        
        qualified.sort(key=lambda c: c.quality_score, reverse=True)
        best = qualified[0]
        
        size = self._calculate_size(best)
        
        if size <= 0:
            return actions
        
        pos = Position(
            symbol=symbol, direction=best.direction,
            entry_price=best.entry, stop_loss=best.stop,
            tp1=best.tp1, tp2=best.tp2, size=size,
            engine=best.engine, entry_bar=bar_idx, ttl_bars=best.ttl_bars
        )
        
        self.positions.append(pos)
        self.last_trade_bar[symbol] = bar_idx
        
        actions.append({
            'action': 'OPEN',
            'symbol': symbol,
            'direction': 'LONG' if best.direction == 1 else 'SHORT',
            'engine': best.engine,
            'entry': best.entry,
            'stop': best.stop,
            'tp1': best.tp1,
            'tp2': best.tp2,
            'size': size,
            'quality': best.quality_score,
            'regime': max(regime.items(), key=lambda x: x[1])
        })
        
        return actions
    
    def _manage_positions(self, symbol: str, candles: pd.DataFrame, bar_idx: int) -> List[dict]:
        """Manage all positions for a symbol."""
        actions = []
        
        if not self.positions:
            return actions
        
        last = candles.iloc[-1]
        high = float(last['high'])
        low = float(last['low'])
        close = float(last['close'])
        
        for pos in self.positions:
            if not pos.active or pos.symbol != symbol:
                continue
            
            bars_held = bar_idx - pos.entry_bar
            if bars_held >= pos.ttl_bars:
                # TTL expired - but only exit if we're significantly underwater
                # If near breakeven, let it continue to ride
                if pos.direction == 1:
                    pnl_pct = (close - pos.entry_price) / pos.entry_price
                    if pnl_pct < -0.005:  # More than 0.5% underwater - stricter
                        self._close_position(pos, close, reason='TTL')
                        actions.append({'action': 'CLOSE', 'symbol': symbol, 'reason': 'TTL', 'price': close})
                        continue
                if pos.direction == -1:
                    pnl_pct = (pos.entry_price - close) / pos.entry_price
                    if pnl_pct < -0.005:  # More than 0.5% underwater
                        self._close_position(pos, close, reason='TTL')
                        actions.append({'action': 'CLOSE', 'symbol': symbol, 'reason': 'TTL', 'price': close})
                        continue
            
            if pos.direction == 1 and low <= pos.stop_loss:
                self._close_position(pos, pos.stop_loss, reason='STOP', is_loss=True)
                actions.append({'action': 'CLOSE', 'symbol': symbol, 'reason': 'STOP', 'price': pos.stop_loss})
                continue
            if pos.direction == -1 and high >= pos.stop_loss:
                self._close_position(pos, pos.stop_loss, reason='STOP', is_loss=True)
                actions.append({'action': 'CLOSE', 'symbol': symbol, 'reason': 'STOP', 'price': pos.stop_loss})
                continue
            
            if not pos.tp1_taken:
                if pos.direction == 1 and high >= pos.tp1:
                    pos.tp1_taken = True
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price)
                    actions.append({'action': 'TP1', 'symbol': symbol, 'new_stop': pos.stop_loss})
                elif pos.direction == -1 and low <= pos.tp1:
                    pos.tp1_taken = True
                    pos.stop_loss = min(pos.stop_loss, pos.entry_price)
                    actions.append({'action': 'TP1', 'symbol': symbol, 'new_stop': pos.stop_loss})
            
            if pos.direction == 1 and high >= pos.tp2:
                self._close_position(pos, pos.tp2, reason='TP2', is_loss=False)
                actions.append({'action': 'CLOSE', 'symbol': symbol, 'reason': 'TP2', 'price': pos.tp2})
                continue
            if pos.direction == -1 and low <= pos.tp2:
                self._close_position(pos, pos.tp2, reason='TP2', is_loss=False)
                actions.append({'action': 'CLOSE', 'symbol': symbol, 'reason': 'TP2', 'price': pos.tp2})
                continue
        
        return actions
    
    def _close_position(self, pos: Position, exit_price: float, reason: str, is_loss: bool = None):
        """Close a position and update equity."""
        if not pos.active:
            return
        
        if pos.direction == 1:
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
        
        total_fee_pct = 2 * self.cfg.fee_pct + self.cfg.slippage_pct
        net_pnl_pct = pnl_pct - total_fee_pct
        
        position_value = pos.size * pos.entry_price
        pnl_dollars = position_value * net_pnl_pct
        self.equity += pnl_dollars
        
        if is_loss is not None:
            self.trade_results.append(0 if is_loss else 1)
        elif net_pnl_pct > 0:
            self.trade_results.append(1)
        else:
            self.trade_results.append(0)
        
        pos.active = False
    
    def _calculate_size(self, candidate: CandidateTrade) -> float:
        """Calculate position size with all brakes applied."""
        risk_dollars = self.equity * self.cfg.base_risk_pct
        
        self.peak_equity = max(self.peak_equity, self.equity)
        dd = (self.peak_equity - self.equity) / self.peak_equity
        dd_mult = self._drawdown_brake(dd)
        
        trust_mult = self._trust_brake()
        quality_mult = 0.8 + (candidate.quality_score * 0.4)
        
        adjusted_risk = risk_dollars * dd_mult * trust_mult * quality_mult
        
        stop_dist = abs(candidate.entry - candidate.stop)
        if stop_dist == 0:
            return 0
        
        size = adjusted_risk / stop_dist
        return size
    
    def _drawdown_brake(self, dd: float) -> float:
        """Reduce size based on drawdown."""
        if dd <= self.cfg.drawdown_brake_start:
            return 1.0
        if dd >= self.cfg.drawdown_brake_max:
            return 0.25
        t = (dd - self.cfg.drawdown_brake_start) / (self.cfg.drawdown_brake_max - self.cfg.drawdown_brake_start)
        return 1.0 - (0.75 * t)
    
    def _trust_brake(self) -> float:
        """Reduce size if recent performance is poor."""
        if len(self.trade_results) < 15:
            return 1.0
        recent = self.trade_results[-self.cfg.trust_window_trades:]
        winrate = sum(recent) / len(recent)
        if winrate >= self.cfg.trust_min_winrate:
            return 1.0
        gap = self.cfg.trust_min_winrate - winrate
        return max(0.4, 1.0 - 2.5 * gap)
    
    def reset_day(self):
        """Call at start of new trading day."""
        self.day_start_equity = self.equity
        self.trades_today = 0


def check_hydra_signal(df: pd.DataFrame, symbol: str, cfg: HydraConfig = None) -> Optional[dict]:
    """Quick signal check for live trading."""
    if cfg is None:
        cfg = HydraConfig()
    
    if len(df) < 150:
        return None
    
    features = FeatureEngine()
    regime_detector = RegimeDetector(cfg)
    engines = AlphaEngines(cfg)
    
    feats = features.compute(df)
    if len(feats) < 100:
        return None
    
    regime = regime_detector.detect(feats)
    if max(regime.values()) < cfg.min_regime_confidence:
        return None
    
    candidates = engines.generate(symbol, df, feats, regime)
    qualified = [c for c in candidates if c.quality_score >= cfg.min_quality_score]
    
    if not qualified:
        return None
    
    qualified.sort(key=lambda c: c.quality_score, reverse=True)
    best = qualified[0]
    
    return {
        'direction': 'LONG' if best.direction == 1 else 'SHORT',
        'engine': best.engine,
        'entry': best.entry,
        'stop': best.stop,
        'tp1': best.tp1,
        'tp2': best.tp2,
        'quality': best.quality_score,
        'regime': max(regime.items(), key=lambda x: x[1]),
        'atr': float(feats.iloc[-1]['atr']) if not pd.isna(feats.iloc[-1]['atr']) else 0
    }


if __name__ == "__main__":
    print("HYDRA-SPOT Strategy")
    print("=" * 50)
    print("\nEngines:")
    for eng in AlphaEngines(HydraConfig()).engine_regimes:
        print(f"  - {eng}")
    print("\nKey features:")
    print("  - Quality gating (only best setups)")
    print("  - TP1 → breakeven stop → TP2")
    print("  - Drawdown + Trust brakes")
    print("  - TTL fast-fail")
