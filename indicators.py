"""
Technical indicators for trading signal generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("indicators")

class TechnicalIndicators:
    """Technical indicators for trading signal generation."""
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = config.RSI_PERIOD) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            
        Returns:
            DataFrame with RSI column added
        """
        if len(df) < period:
            df['rsi'] = np.nan
            return df
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame, 
        period: int = config.BOLLINGER_PERIOD, 
        std: float = config.BOLLINGER_STD
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            period: Bollinger Bands period
            std: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        if len(df) < period:
            df['bb_middle'] = np.nan
            df['bb_upper'] = np.nan
            df['bb_lower'] = np.nan
            df['bb_width'] = np.nan
            return df
        
        # Calculate middle band (SMA)
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std)
        
        # Calculate bandwidth
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    @staticmethod
    def add_macd(
        df: pd.DataFrame, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with MACD columns added
        """
        if len(df) < slow_period + signal_period:
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
            df['macd_histogram'] = np.nan
            return df
        
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['macd'] = ema_fast - ema_slow
        
        # Calculate signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Add volume-based indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for volume indicators
            
        Returns:
            DataFrame with volume indicator columns added
        """
        if len(df) < period:
            df['volume_sma'] = np.nan
            df['volume_ratio'] = np.nan
            return df
        
        # Calculate volume SMA
        df['volume_sma'] = df['volume'].rolling(window=period).mean()
        
        # Calculate volume ratio (current volume / average volume)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    @staticmethod
    def add_stochastic_oscillator(
        df: pd.DataFrame, 
        k_period: int = 14, 
        d_period: int = 3
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            k_period: %K period
            d_period: %D period
            
        Returns:
            DataFrame with Stochastic Oscillator columns added
        """
        if len(df) < k_period:
            df['stoch_k'] = np.nan
            df['stoch_d'] = np.nan
            return df
        
        # Calculate %K
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D (SMA of %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR) to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            DataFrame with ATR column added
        """
        if len(df) < period + 1:
            df['atr'] = np.nan
            return df
        
        # Calculate true range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Drop temporary column
        df.drop('tr', axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicator columns added
        """
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_volume_indicators(df)
        df = TechnicalIndicators.add_stochastic_oscillator(df)
        df = TechnicalIndicators.add_atr(df)
        
        return df
    
    @staticmethod
    def generate_signals(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Dictionary with signal information
        """
        if df.empty or df.iloc[-1].isna().any():
            return {
                'buy_signal': False,
                'sell_signal': False,
                'signal_strength': 0,
                'reasons': []
            }
        
        # Get the latest values
        latest = df.iloc[-1]
        
        # Initialize signals
        buy_signals = []
        sell_signals = []
        
        # RSI signals
        if latest['rsi'] < config.RSI_OVERSOLD:
            buy_signals.append(f"RSI oversold ({latest['rsi']:.2f})")
        elif latest['rsi'] > config.RSI_OVERBOUGHT:
            sell_signals.append(f"RSI overbought ({latest['rsi']:.2f})")
        
        # Bollinger Bands signals
        if latest['close'] < latest['bb_lower']:
            buy_signals.append("Price below lower Bollinger Band")
        elif latest['close'] > latest['bb_upper']:
            sell_signals.append("Price above upper Bollinger Band")
        
        # MACD signals
        if latest['macd'] > latest['macd_signal'] and df.iloc[-2]['macd'] <= df.iloc[-2]['macd_signal']:
            buy_signals.append("MACD crossed above signal line")
        elif latest['macd'] < latest['macd_signal'] and df.iloc[-2]['macd'] >= df.iloc[-2]['macd_signal']:
            sell_signals.append("MACD crossed below signal line")
        
        # Volume signals
        if latest['volume_ratio'] > 1.5:
            if latest['close'] > df.iloc[-2]['close']:
                buy_signals.append(f"High volume price increase (volume ratio: {latest['volume_ratio']:.2f})")
            elif latest['close'] < df.iloc[-2]['close']:
                sell_signals.append(f"High volume price decrease (volume ratio: {latest['volume_ratio']:.2f})")
        
        # Stochastic signals
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20 and latest['stoch_k'] > latest['stoch_d']:
            buy_signals.append("Stochastic oversold with bullish crossover")
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80 and latest['stoch_k'] < latest['stoch_d']:
            sell_signals.append("Stochastic overbought with bearish crossover")
        
        # Calculate signal strength (0-100)
        buy_strength = min(100, len(buy_signals) * 25)
        sell_strength = min(100, len(sell_signals) * 25)
        
        # Determine final signal
        buy_signal = buy_strength >= 50
        sell_signal = sell_strength >= 50
        
        # If both signals are present, use the stronger one
        if buy_signal and sell_signal:
            if buy_strength > sell_strength:
                sell_signal = False
            else:
                buy_signal = False
        
        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': buy_strength if buy_signal else (sell_strength if sell_signal else 0),
            'buy_reasons': buy_signals,
            'sell_reasons': sell_signals
        }
