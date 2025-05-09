"""
Coin selector module for identifying volatile coins suitable for trading.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

import config
from bybit_api import BybitAPI

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("coin_selector")

class CoinSelector:
    """Selects coins for trading based on volatility and other metrics."""
    
    def __init__(self, api: BybitAPI):
        """
        Initialize the coin selector.
        
        Args:
            api: Initialized BybitAPI instance
        """
        self.api = api
        logger.info("Coin selector initialized")
    
    def calculate_volatility(self, klines: pd.DataFrame) -> float:
        """
        Calculate volatility based on price movements.
        
        Args:
            klines: DataFrame with OHLCV data
            
        Returns:
            Volatility score
        """
        if klines.empty:
            return 0.0
        
        # Calculate daily returns
        returns = klines['close'].pct_change().dropna()
        
        # Calculate volatility (standard deviation of returns)
        volatility = returns.std()
        
        # Calculate average true range (ATR)
        high_low = klines['high'] - klines['low']
        high_close = (klines['high'] - klines['close'].shift()).abs()
        low_close = (klines['low'] - klines['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1).dropna()
        atr = true_range.mean() / klines['close'].iloc[-1]
        
        # Combine metrics for a volatility score
        volatility_score = (volatility * 0.7) + (atr * 0.3)
        
        return float(volatility_score)
    
    def calculate_volume_stability(self, klines: pd.DataFrame) -> float:
        """
        Calculate volume stability score.
        
        Args:
            klines: DataFrame with OHLCV data
            
        Returns:
            Volume stability score (0-1)
        """
        if klines.empty or len(klines) < 10:
            return 0.0
        
        # Calculate coefficient of variation for volume
        volume_mean = klines['volume'].mean()
        volume_std = klines['volume'].std()
        
        if volume_mean == 0:
            return 0.0
        
        # Lower coefficient of variation means more stable volume
        cv = volume_std / volume_mean
        
        # Convert to a 0-1 score (lower CV is better)
        stability_score = max(0, min(1, 1 - (cv / 2)))
        
        return float(stability_score)
    
    def calculate_trend_strength(self, klines: pd.DataFrame) -> Tuple[float, str]:
        """
        Calculate the strength and direction of the current trend.
        
        Args:
            klines: DataFrame with OHLCV data
            
        Returns:
            Tuple of (trend_strength, trend_direction)
        """
        if klines.empty or len(klines) < 20:
            return 0.0, "neutral"
        
        # Simple moving averages
        short_ma = klines['close'].rolling(window=10).mean().iloc[-1]
        medium_ma = klines['close'].rolling(window=20).mean().iloc[-1]
        long_ma = klines['close'].rolling(window=50).mean().iloc[-1]
        
        # Determine trend direction
        if short_ma > medium_ma > long_ma:
            trend_direction = "bullish"
        elif short_ma < medium_ma < long_ma:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"
        
        # Calculate trend strength based on price distance from moving averages
        current_price = klines['close'].iloc[-1]
        
        # Normalize distances
        short_distance = abs(current_price - short_ma) / current_price
        medium_distance = abs(current_price - medium_ma) / current_price
        long_distance = abs(current_price - long_ma) / current_price
        
        # Average the distances for a strength score
        trend_strength = (short_distance + medium_distance + long_distance) / 3
        
        # Cap at 1.0
        trend_strength = min(1.0, trend_strength * 10)
        
        return float(trend_strength), trend_direction
    
    def calculate_spread(self, market_data: Dict) -> float:
        """
        Calculate the bid-ask spread percentage.
        
        Args:
            market_data: Market data from Bybit API
            
        Returns:
            Spread as a percentage
        """
        if not market_data or 'bid1Price' not in market_data or 'ask1Price' not in market_data:
            return float('inf')
        
        bid = float(market_data['bid1Price'])
        ask = float(market_data['ask1Price'])
        
        if bid == 0:
            return float('inf')
        
        spread = (ask - bid) / bid
        
        return float(spread)
    
    def score_coin(self, symbol: str) -> Dict:
        """
        Calculate a comprehensive score for a coin.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            Dictionary with coin metrics and score
        """
        # Get kline data for different timeframes
        klines_1h = self.api.get_klines(symbol, '1h', 100)
        klines_5m = self.api.get_klines(symbol, '5m', 100)
        
        if klines_1h.empty or klines_5m.empty:
            logger.warning(f"Insufficient data for {symbol}")
            return {
                'symbol': symbol,
                'score': 0,
                'valid': False
            }
        
        # Get current market data
        market_data = self.api.get_market_data(symbol)
        
        if not market_data:
            logger.warning(f"No market data for {symbol}")
            return {
                'symbol': symbol,
                'score': 0,
                'valid': False
            }
        
        # Calculate metrics
        volatility = self.calculate_volatility(klines_1h)
        volume_stability = self.calculate_volume_stability(klines_1h)
        trend_strength, trend_direction = self.calculate_trend_strength(klines_1h)
        spread = self.calculate_spread(market_data)
        
        # Check if the coin meets minimum requirements
        volume_24h = float(market_data.get('turnover24h', 0))
        valid = (
            volume_24h >= config.MIN_24H_VOLUME and
            volatility >= config.MIN_VOLATILITY and
            spread <= config.MAX_SPREAD
        )
        
        # Calculate final score (0-100)
        volatility_score = min(1.0, volatility * 10) * 40  # 40% weight
        volume_score = volume_stability * 30  # 30% weight
        trend_score = trend_strength * 20  # 20% weight
        spread_score = (1 - min(1.0, spread * 100)) * 10  # 10% weight
        
        total_score = volatility_score + volume_score + trend_score + spread_score
        
        return {
            'symbol': symbol,
            'score': total_score,
            'volatility': volatility,
            'volume_stability': volume_stability,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'spread': spread,
            'volume_24h': volume_24h,
            'last_price': float(market_data.get('lastPrice', 0)),
            'valid': valid
        }
    
    def select_coins(self, num_coins: int = config.MAX_COINS) -> List[Dict]:
        """
        Select the best coins for trading based on scoring.
        
        Args:
            num_coins: Number of coins to select
            
        Returns:
            List of selected coin data dictionaries
        """
        logger.info(f"Scanning market for top {num_coins} coins...")
        
        # Get all available tickers
        all_tickers = self.api.get_all_tickers()
        
        # Filter for USDT pairs only
        usdt_symbols = [ticker['symbol'] for ticker in all_tickers if ticker['symbol'].endswith('USDT')]
        
        # Score each coin
        coin_scores = []
        for symbol in usdt_symbols:
            try:
                score_data = self.score_coin(symbol)
                if score_data['valid']:
                    coin_scores.append(score_data)
                    logger.debug(f"Scored {symbol}: {score_data['score']:.2f}")
            except Exception as e:
                logger.error(f"Error scoring {symbol}: {e}")
        
        # Sort by score (descending)
        coin_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top coins
        selected_coins = coin_scores[:num_coins]
        
        if selected_coins:
            logger.info(f"Selected {len(selected_coins)} coins: {[c['symbol'] for c in selected_coins]}")
        else:
            logger.warning("No suitable coins found")
        
        return selected_coins
