"""
Bybit API integration for the crypto trading bot.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import pandas as pd

import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bybit_api")

# Load environment variables
load_dotenv()

class BybitAPI:
    """Interface for Bybit exchange API."""
    
    def __init__(self):
        """Initialize the Bybit API client."""
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Bybit API credentials not found in .env file")
        
        self.client = HTTP(
            testnet=config.BYBIT_TESTNET,
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        
        self.last_api_call = 0
        logger.info(f"Bybit API initialized (testnet: {config.BYBIT_TESTNET})")
    
    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < 1 / config.API_RATE_LIMIT:
            sleep_time = (1 / config.API_RATE_LIMIT) - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def get_account_balance(self) -> float:
        """Get the current account balance in USD."""
        self._rate_limit()
        try:
            response = self.client.get_wallet_balance(accountType="UNIFIED")
            if response["retCode"] == 0:
                # Extract USDT balance from the response
                for coin in response["result"]["list"][0]["coin"]:
                    if coin["coin"] == "USDT":
                        return float(coin["walletBalance"])
            
            logger.error(f"Failed to get account balance: {response}")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0
    
    def get_all_tickers(self) -> List[Dict]:
        """Get all available tickers from Bybit."""
        self._rate_limit()
        try:
            response = self.client.get_tickers(category="spot")
            if response["retCode"] == 0:
                return response["result"]["list"]
            
            logger.error(f"Failed to get tickers: {response}")
            return []
        except Exception as e:
            logger.error(f"Error getting tickers: {e}")
            return []
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """
        Get candlestick data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert interval format to what Bybit API expects
        # Map common time intervals to Bybit's expected format
        interval_map = {
            '1m': '1', 
            '3m': '3', 
            '5m': '5',
            '15m': '15',
            '30m': '30',
            '1h': '60',
            '2h': '120',
            '4h': '240',
            '6h': '360',
            '12h': '720',
            '1d': 'D',
            '1w': 'W',
            '1M': 'M'
        }
        
        # Convert the interval if it's in our map
        if interval in interval_map:
            interval = interval_map[interval]
        
        self._rate_limit()
        try:
            response = self.client.get_kline(
                category="spot",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if response["retCode"] == 0:
                # Convert the kline data to a DataFrame
                df = pd.DataFrame(
                    response["result"]["list"],
                    columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"]
                )
                
                # Convert types
                for col in ["open", "high", "low", "close", "volume", "turnover"]:
                    df[col] = pd.to_numeric(df[col])
                
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                
                # Reverse the order to have the oldest data first
                df = df.iloc[::-1].reset_index(drop=True)
                
                return df
            
            logger.error(f"Failed to get klines for {symbol}: {response}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def place_order(
        self, 
        symbol: str, 
        side: str, 
        order_type: str, 
        qty: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Place an order on Bybit.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'Buy' or 'Sell'
            order_type: 'Limit' or 'Market'
            qty: Order quantity
            price: Order price (required for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order response from Bybit
        """
        self._rate_limit()
        try:
            params = {
                "category": "spot",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
            }
            
            if order_type == "Limit" and price is not None:
                params["price"] = str(price)
            
            if stop_loss is not None:
                params["stopLoss"] = str(stop_loss)
                
            if take_profit is not None:
                params["takeProfit"] = str(take_profit)
            
            response = self.client.place_order(**params)
            
            if response["retCode"] == 0:
                logger.info(f"Order placed successfully: {symbol} {side} {qty}")
                return response["result"]
            
            logger.error(f"Failed to place order: {response}")
            return {}
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders, optionally filtered by symbol."""
        self._rate_limit()
        try:
            params = {"category": "spot"}
            if symbol:
                params["symbol"] = symbol
                
            response = self.client.get_open_orders(**params)
            
            if response["retCode"] == 0:
                return response["result"]["list"]
            
            logger.error(f"Failed to get open orders: {response}")
            return []
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order by its ID."""
        self._rate_limit()
        try:
            response = self.client.cancel_order(
                category="spot",
                symbol=symbol,
                orderId=order_id
            )
            
            if response["retCode"] == 0:
                logger.info(f"Order {order_id} cancelled successfully")
                return True
            
            logger.error(f"Failed to cancel order {order_id}: {response}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get order history, optionally filtered by symbol."""
        self._rate_limit()
        try:
            params = {
                "category": "spot",
                "limit": limit
            }
            if symbol:
                params["symbol"] = symbol
                
            response = self.client.get_order_history(**params)
            
            if response["retCode"] == 0:
                return response["result"]["list"]
            
            logger.error(f"Failed to get order history: {response}")
            return []
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return []
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for a symbol."""
        self._rate_limit()
        try:
            response = self.client.get_tickers(
                category="spot",
                symbol=symbol
            )
            
            if response["retCode"] == 0 and response["result"]["list"]:
                return response["result"]["list"][0]
            
            logger.error(f"Failed to get market data for {symbol}: {response}")
            return {}
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}
