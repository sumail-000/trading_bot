"""
Main entry point for the crypto trading bot.
"""

import os
import time
import logging
import threading
import schedule
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

import config
from bybit_api import BybitAPI
from coin_selector import CoinSelector
from risk_manager import RiskManager
from trading_engine import TradingEngine

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

class TradingBot:
    """Main trading bot class that coordinates all components."""
    
    def __init__(self):
        """Initialize the trading bot."""
        # Load environment variables
        load_dotenv()
        
        # Initialize API
        self.api = BybitAPI()
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.api)
        
        # Initialize coin selector
        self.coin_selector = CoinSelector(self.api)
        
        # Trading engines for each coin
        self.trading_engines = {}
        
        # Bot status
        self.running = False
        self.start_time = None
        
        logger.info("Trading bot initialized")
    
    def start(self) -> None:
        """Start the trading bot."""
        if self.running:
            logger.warning("Trading bot already running")
            return
        
        logger.info("Starting trading bot...")
        self.running = True
        self.start_time = datetime.utcnow()
        
        # Select coins for trading
        self._select_coins()
        
        # Set up scheduled tasks
        schedule.every(4).hours.do(self._select_coins)
        schedule.every(1).minutes.do(self._log_status)
        
        # Start scheduler in a separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Trading bot started successfully")
    
    def stop(self) -> None:
        """Stop the trading bot."""
        if not self.running:
            logger.warning("Trading bot already stopped")
            return
        
        logger.info("Stopping trading bot...")
        self.running = False
        
        # Stop all trading engines
        for engine in self.trading_engines.values():
            engine.stop()
        
        self.trading_engines = {}
        
        # Clear scheduled jobs
        schedule.clear()
        
        # Wait for scheduler thread to finish
        if hasattr(self, 'scheduler_thread') and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Trading bot stopped")
    
    def _run_scheduler(self) -> None:
        """Run the scheduler loop."""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def _select_coins(self) -> None:
        """Select coins for trading and create/update trading engines."""
        logger.info("Selecting coins for trading...")
        
        # Get top coins
        selected_coins = self.coin_selector.select_coins(num_coins=config.MAX_COINS)
        
        if not selected_coins:
            logger.warning("No suitable coins found for trading")
            return
        
        # Stop engines for coins that are no longer selected
        current_symbols = set(self.trading_engines.keys())
        new_symbols = set(coin['symbol'] for coin in selected_coins)
        
        # Symbols to remove
        for symbol in current_symbols - new_symbols:
            logger.info(f"Removing trading engine for {symbol}")
            self.trading_engines[symbol].stop()
            del self.trading_engines[symbol]
        
        # Create or update engines for selected coins
        for coin_data in selected_coins:
            symbol = coin_data['symbol']
            
            if symbol in self.trading_engines:
                # Update existing engine
                logger.debug(f"Trading engine for {symbol} already exists")
                continue
            
            # Create new engine
            logger.info(f"Creating trading engine for {symbol}")
            engine = TradingEngine(
                api=self.api,
                risk_manager=self.risk_manager,
                symbol=symbol,
                coin_data=coin_data
            )
            
            # Start the engine
            engine.start()
            
            # Add to engines dictionary
            self.trading_engines[symbol] = engine
        
        logger.info(f"Selected {len(selected_coins)} coins for trading: {[c['symbol'] for c in selected_coins]}")
    
    def _log_status(self) -> None:
        """Log the current status of the trading bot."""
        if not self.running:
            return
        
        # Get risk manager status
        risk_status = self.risk_manager.get_trading_status()
        
        # Get engine statuses
        engine_statuses = {symbol: engine.get_status() for symbol, engine in self.trading_engines.items()}
        
        # Log summary
        active_trades = sum(1 for status in engine_statuses.values() if status['active_trade'])
        total_trades = sum(status['total_trades'] for status in engine_statuses.values())
        total_profit = sum(status['total_profit'] for status in engine_statuses.values())
        
        logger.info(f"Status: Balance=${risk_status['current_balance']:.2f}, "
                  f"Daily profit=${risk_status['daily_profit']:.2f}, "
                  f"Active trades={active_trades}, "
                  f"Total trades={total_trades}, "
                  f"Total profit=${total_profit:.2f}")
        
        # Log detailed engine status if there are active trades
        for symbol, status in engine_statuses.items():
            if status['active_trade']:
                trade = status['active_trade']
                side = trade['side']
                entry_price = trade['entry_price']
                current_price = self.api.get_market_data(symbol).get('lastPrice', 0)
                
                if side == 'Buy':
                    pnl_pct = (float(current_price) - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - float(current_price)) / entry_price * 100
                
                logger.info(f"Active trade {symbol} {side}: Entry=${entry_price:.6f}, "
                          f"Current=${float(current_price):.6f}, "
                          f"P/L={pnl_pct:.2f}%")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the trading bot.
        
        Returns:
            Dictionary with bot status information
        """
        if not self.running:
            return {'running': False}
        
        # Get risk manager status
        risk_status = self.risk_manager.get_trading_status()
        
        # Get engine statuses
        engine_statuses = {symbol: engine.get_status() for symbol, engine in self.trading_engines.items()}
        
        # Calculate summary statistics
        active_trades = sum(1 for status in engine_statuses.values() if status['active_trade'])
        total_trades = sum(status['total_trades'] for status in engine_statuses.values())
        winning_trades = sum(status['winning_trades'] for status in engine_statuses.values())
        losing_trades = sum(status['losing_trades'] for status in engine_statuses.values())
        win_rate = winning_trades / max(1, total_trades) * 100
        total_profit = sum(status['total_profit'] for status in engine_statuses.values())
        
        return {
            'running': self.running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            'balance': risk_status['current_balance'],
            'initial_balance': risk_status['initial_balance'],
            'daily_profit': risk_status['daily_profit'],
            'daily_profit_target': risk_status['daily_profit_target'],
            'active_trades': active_trades,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'trading_engines': list(self.trading_engines.keys()),
            'engine_statuses': engine_statuses
        }


def main():
    """Main entry point for the trading bot."""
    try:
        # Create and start the trading bot
        bot = TradingBot()
        bot.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        if 'bot' in locals():
            bot.stop()
    
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        if 'bot' in locals():
            bot.stop()


if __name__ == "__main__":
    main()
