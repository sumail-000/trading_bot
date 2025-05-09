"""
Risk manager module for capital allocation and risk control.
"""

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import config
from bybit_api import BybitAPI

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("risk_manager")

class RiskManager:
    """Manages capital allocation and risk control for the trading bot."""
    
    def __init__(self, api: BybitAPI):
        """
        Initialize the risk manager.
        
        Args:
            api: Initialized BybitAPI instance
        """
        self.api = api
        self.initial_balance = config.INITIAL_CAPITAL
        self.current_balance = self.api.get_account_balance()
        self.daily_profit = 0.0
        self.trades_today = 0
        self.winning_trades_today = 0
        self.losing_trades_today = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.daily_reset_time = self._get_next_reset_time()
        
        # Allocate capital to each coin
        self.coin_allocations = {}
        self.update_allocations()
        
        logger.info(f"Risk manager initialized with balance: ${self.current_balance:.2f}")
    
    def _get_next_reset_time(self) -> datetime:
        """Get the next daily reset time (midnight UTC)."""
        now = datetime.utcnow()
        tomorrow = now + timedelta(days=1)
        return datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0)
    
    def update_allocations(self) -> Dict[str, float]:
        """
        Update capital allocations for each coin.
        
        Returns:
            Dictionary mapping coin symbols to their allocation amount
        """
        # Get current balance
        self.current_balance = self.api.get_account_balance()
        
        # Divide capital equally among coins
        per_coin_allocation = self.current_balance / config.MAX_COINS
        
        # Create allocation dictionary
        self.coin_allocations = {}
        
        logger.info(f"Updated allocations: ${per_coin_allocation:.4f} per coin")
        
        return self.coin_allocations
    
    def get_position_size(self, symbol: str, entry_price: float) -> float:
        """
        Calculate the position size for a trade.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the trade
            
        Returns:
            Position size in base currency
        """
        # Get allocation for this coin
        allocation = self.current_balance / config.MAX_COINS
        
        # Calculate trade size based on allocation percentage
        trade_amount = allocation * config.TRADE_SIZE_PERCENTAGE
        
        # Calculate quantity based on entry price
        quantity = trade_amount / entry_price
        
        # Round to appropriate precision (this is a simplified approach)
        # In a real implementation, you would get the precision from the exchange
        precision = 6 if entry_price < 1 else (4 if entry_price < 100 else 2)
        quantity = round(quantity, precision)
        
        logger.debug(f"Position size for {symbol}: {quantity} (${trade_amount:.4f})")
        
        return quantity
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """
        Calculate stop loss price for a trade.
        
        Args:
            entry_price: Entry price for the trade
            side: 'Buy' or 'Sell'
            
        Returns:
            Stop loss price
        """
        if side == 'Buy':
            stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENTAGE / 100)
        else:
            stop_loss = entry_price * (1 + config.STOP_LOSS_PERCENTAGE / 100)
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """
        Calculate take profit price for a trade.
        
        Args:
            entry_price: Entry price for the trade
            side: 'Buy' or 'Sell'
            
        Returns:
            Take profit price
        """
        if side == 'Buy':
            take_profit = entry_price * (1 + config.TAKE_PROFIT_PERCENTAGE / 100)
        else:
            take_profit = entry_price * (1 - config.TAKE_PROFIT_PERCENTAGE / 100)
        
        return take_profit
    
    def update_trade_stats(self, profit: float) -> None:
        """
        Update trading statistics after a trade is completed.
        
        Args:
            profit: Profit/loss from the trade in USD
        """
        self.trades_today += 1
        self.daily_profit += profit
        
        if profit > 0:
            self.winning_trades_today += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades_today += 1
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        logger.info(f"Trade completed: ${profit:.4f} profit, daily total: ${self.daily_profit:.4f}")
    
    def check_daily_reset(self) -> bool:
        """
        Check if it's time to reset daily statistics.
        
        Returns:
            True if stats were reset, False otherwise
        """
        now = datetime.utcnow()
        
        if now >= self.daily_reset_time:
            # Reset daily stats
            logger.info(f"Daily reset - Previous stats: Profit=${self.daily_profit:.2f}, "
                      f"Trades={self.trades_today}, Win/Loss={self.winning_trades_today}/{self.losing_trades_today}")
            
            self.daily_profit = 0.0
            self.trades_today = 0
            self.winning_trades_today = 0
            self.losing_trades_today = 0
            self.daily_reset_time = self._get_next_reset_time()
            
            # Update allocations
            self.update_allocations()
            
            return True
        
        return False
    
    def should_stop_trading(self) -> bool:
        """
        Determine if trading should be stopped for the day.
        
        Returns:
            True if trading should stop, False otherwise
        """
        # Stop if daily profit target is reached
        if self.daily_profit >= config.DAILY_PROFIT_TARGET:
            logger.info(f"Daily profit target reached: ${self.daily_profit:.2f} >= ${config.DAILY_PROFIT_TARGET:.2f}")
            return True
        
        # Stop if too many consecutive losses
        if self.consecutive_losses >= 5:
            logger.warning(f"Too many consecutive losses: {self.consecutive_losses}")
            return True
        
        # Stop if lost more than 10% of initial capital
        if self.current_balance < self.initial_balance * 0.9:
            logger.warning(f"Capital drawdown limit reached: ${self.current_balance:.2f} < ${self.initial_balance * 0.9:.2f}")
            return True
        
        return False
    
    def adjust_parameters_based_on_performance(self) -> None:
        """Adjust trading parameters based on recent performance."""
        win_rate = self.winning_trades_today / max(1, self.trades_today)
        
        # If win rate is low, reduce risk
        if win_rate < 0.4 and self.trades_today >= 5:
            # Reduce trade size
            config.TRADE_SIZE_PERCENTAGE = max(0.05, config.TRADE_SIZE_PERCENTAGE * 0.8)
            
            # Increase take profit (aim for smaller but more certain profits)
            config.TAKE_PROFIT_PERCENTAGE = min(2.0, config.TAKE_PROFIT_PERCENTAGE * 1.2)
            
            logger.info(f"Adjusted parameters due to low win rate: "
                      f"Trade size={config.TRADE_SIZE_PERCENTAGE:.2f}, "
                      f"Take profit={config.TAKE_PROFIT_PERCENTAGE:.2f}%")
        
        # If win rate is high, can slightly increase risk
        elif win_rate > 0.7 and self.trades_today >= 5:
            # Increase trade size slightly
            config.TRADE_SIZE_PERCENTAGE = min(0.2, config.TRADE_SIZE_PERCENTAGE * 1.1)
            
            logger.info(f"Adjusted parameters due to high win rate: "
                      f"Trade size={config.TRADE_SIZE_PERCENTAGE:.2f}")
    
    def get_trading_status(self) -> Dict:
        """
        Get current trading status and statistics.
        
        Returns:
            Dictionary with trading status information
        """
        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'daily_profit': self.daily_profit,
            'daily_profit_target': config.DAILY_PROFIT_TARGET,
            'trades_today': self.trades_today,
            'winning_trades': self.winning_trades_today,
            'losing_trades': self.losing_trades_today,
            'win_rate': self.winning_trades_today / max(1, self.trades_today),
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'should_stop_trading': self.should_stop_trading(),
            'next_reset': self.daily_reset_time.isoformat()
        }
