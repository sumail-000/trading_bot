"""
Trading engine for executing trades on a single coin.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

import config
from bybit_api import BybitAPI
from indicators import TechnicalIndicators
from risk_manager import RiskManager

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trading_engine")

class TradingEngine:
    """Trading engine for a single coin."""
    
    def __init__(
        self, 
        api: BybitAPI, 
        risk_manager: RiskManager, 
        symbol: str,
        coin_data: Dict
    ):
        """
        Initialize the trading engine.
        
        Args:
            api: Initialized BybitAPI instance
            risk_manager: Initialized RiskManager instance
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            coin_data: Coin data from coin selector
        """
        self.api = api
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.coin_data = coin_data
        self.active_trade = None
        self.last_analysis_time = 0
        self.analysis_interval = 60  # seconds
        self.running = False
        self.thread = None
        
        # Trading statistics
        self.trades_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        
        logger.info(f"Trading engine initialized for {symbol}")
    
    def start(self) -> None:
        """Start the trading engine in a separate thread."""
        if self.running:
            logger.warning(f"{self.symbol} trading engine already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._trading_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Started trading engine for {self.symbol}")
    
    def stop(self) -> None:
        """Stop the trading engine."""
        if not self.running:
            logger.warning(f"{self.symbol} trading engine already stopped")
            return
        
        logger.info(f"Stopping trading engine for {self.symbol}")
        self.running = False
        
        # Close any active trades
        if self.active_trade:
            self._close_trade("Trading engine stopped")
        
        if self.thread:
            self.thread.join(timeout=5)
    
    def _trading_loop(self) -> None:
        """Main trading loop."""
        while self.running:
            try:
                # Check if trading should be stopped
                if self.risk_manager.should_stop_trading():
                    logger.info(f"{self.symbol} trading stopped due to risk management")
                    time.sleep(60)
                    continue
                
                # Check if daily reset is needed
                self.risk_manager.check_daily_reset()
                
                # If we have an active trade, monitor it
                if self.active_trade:
                    self._monitor_trade()
                else:
                    # Otherwise, look for new trading opportunities
                    current_time = time.time()
                    if current_time - self.last_analysis_time >= self.analysis_interval:
                        self._analyze_market()
                        self.last_analysis_time = current_time
                
                # Sleep to avoid excessive CPU usage
                time.sleep(5)
            
            except Exception as e:
                logger.error(f"Error in trading loop for {self.symbol}: {e}")
                time.sleep(30)  # Sleep longer on error
    
    def _analyze_market(self) -> None:
        """Analyze the market and look for trading opportunities."""
        # Get market data
        klines = self.api.get_klines(
            symbol=self.symbol,
            interval=config.ANALYSIS_TIMEFRAME,
            limit=100
        )
        
        if klines.empty:
            logger.warning(f"No kline data available for {self.symbol}")
            return
        
        # Add technical indicators
        df = TechnicalIndicators.add_all_indicators(klines)
        
        # Generate signals
        signals = TechnicalIndicators.generate_signals(df)
        
        # Log analysis results
        logger.debug(f"{self.symbol} analysis: Buy={signals['buy_signal']}, Sell={signals['sell_signal']}, "
                   f"Strength={signals['signal_strength']}")
        
        # Execute trade if signal is strong enough
        if signals['buy_signal'] and signals['signal_strength'] >= 75:
            reasons = ", ".join(signals['buy_reasons'])
            logger.info(f"{self.symbol} BUY signal detected: {reasons}")
            self._execute_trade("Buy", reasons)
        
        elif signals['sell_signal'] and signals['signal_strength'] >= 75:
            reasons = ", ".join(signals['sell_reasons'])
            logger.info(f"{self.symbol} SELL signal detected: {reasons}")
            self._execute_trade("Sell", reasons)
    
    def _execute_trade(self, side: str, reasons: str) -> bool:
        """
        Execute a trade based on the signal.
        
        Args:
            side: 'Buy' or 'Sell'
            reasons: Reasons for the trade
            
        Returns:
            True if trade was executed, False otherwise
        """
        if self.active_trade:
            logger.warning(f"Cannot execute {side} trade for {self.symbol}: Active trade exists")
            return False
        
        # Get current market data
        market_data = self.api.get_market_data(self.symbol)
        if not market_data:
            logger.error(f"Cannot execute trade: No market data for {self.symbol}")
            return False
        
        # Get entry price
        entry_price = float(market_data['lastPrice'])
        
        # Calculate position size
        quantity = self.risk_manager.get_position_size(self.symbol, entry_price)
        
        if quantity <= 0:
            logger.warning(f"Cannot execute trade: Invalid position size for {self.symbol}")
            return False
        
        # Calculate stop loss and take profit
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, side)
        take_profit = self.risk_manager.calculate_take_profit(entry_price, side)
        
        # Place order
        order = self.api.place_order(
            symbol=self.symbol,
            side=side,
            order_type="Market",
            qty=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if not order:
            logger.error(f"Failed to place {side} order for {self.symbol}")
            return False
        
        # Record active trade
        self.active_trade = {
            'order_id': order.get('orderId'),
            'symbol': self.symbol,
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.utcnow().isoformat(),
            'reasons': reasons,
            'status': 'open'
        }
        
        logger.info(f"Executed {side} trade for {self.symbol}: {quantity} @ ${entry_price:.6f}, "
                  f"SL=${stop_loss:.6f}, TP=${take_profit:.6f}")
        
        return True
    
    def _monitor_trade(self) -> None:
        """Monitor the active trade and update its status."""
        if not self.active_trade:
            return
        
        # Get current market data
        market_data = self.api.get_market_data(self.symbol)
        if not market_data:
            logger.warning(f"Cannot monitor trade: No market data for {self.symbol}")
            return
        
        current_price = float(market_data['lastPrice'])
        
        # Check if stop loss or take profit has been hit
        side = self.active_trade['side']
        entry_price = self.active_trade['entry_price']
        stop_loss = self.active_trade['stop_loss']
        take_profit = self.active_trade['take_profit']
        
        # Calculate current profit/loss
        if side == 'Buy':
            price_change = current_price - entry_price
            price_change_pct = (price_change / entry_price) * 100
            
            if current_price <= stop_loss:
                self._close_trade(f"Stop loss hit: ${current_price:.6f} <= ${stop_loss:.6f}")
            elif current_price >= take_profit:
                self._close_trade(f"Take profit hit: ${current_price:.6f} >= ${take_profit:.6f}")
        else:  # Sell
            price_change = entry_price - current_price
            price_change_pct = (price_change / entry_price) * 100
            
            if current_price >= stop_loss:
                self._close_trade(f"Stop loss hit: ${current_price:.6f} >= ${stop_loss:.6f}")
            elif current_price <= take_profit:
                self._close_trade(f"Take profit hit: ${current_price:.6f} <= ${take_profit:.6f}")
        
        # Log current trade status
        logger.debug(f"{self.symbol} trade status: {price_change_pct:.2f}% P/L")
    
    def _close_trade(self, reason: str) -> None:
        """
        Close the active trade.
        
        Args:
            reason: Reason for closing the trade
        """
        if not self.active_trade:
            return
        
        # Get current market data
        market_data = self.api.get_market_data(self.symbol)
        if not market_data:
            logger.warning(f"Cannot close trade: No market data for {self.symbol}")
            return
        
        exit_price = float(market_data['lastPrice'])
        
        # Calculate profit/loss
        side = self.active_trade['side']
        entry_price = self.active_trade['entry_price']
        quantity = self.active_trade['quantity']
        
        if side == 'Buy':
            price_change = exit_price - entry_price
        else:  # Sell
            price_change = entry_price - exit_price
        
        profit_amount = price_change * quantity
        profit_percentage = (price_change / entry_price) * 100
        
        # Update trade record
        self.active_trade['exit_price'] = exit_price
        self.active_trade['exit_time'] = datetime.utcnow().isoformat()
        self.active_trade['profit_amount'] = profit_amount
        self.active_trade['profit_percentage'] = profit_percentage
        self.active_trade['close_reason'] = reason
        self.active_trade['status'] = 'closed'
        
        # Update statistics
        self.total_trades += 1
        self.total_profit += profit_amount
        
        if profit_amount > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Add to trade history
        self.trades_history.append(self.active_trade)
        
        # Update risk manager
        self.risk_manager.update_trade_stats(profit_amount)
        
        # Log trade closure
        logger.info(f"Closed {side} trade for {self.symbol}: {reason}, "
                  f"P/L=${profit_amount:.4f} ({profit_percentage:.2f}%)")
        
        # Clear active trade
        self.active_trade = None
    
    def get_status(self) -> Dict:
        """
        Get the current status of the trading engine.
        
        Returns:
            Dictionary with trading engine status
        """
        return {
            'symbol': self.symbol,
            'active_trade': self.active_trade,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades) * 100,
            'total_profit': self.total_profit,
            'running': self.running
        }
