"""
Configuration settings for the crypto trading bot.
"""

# General settings
INITIAL_CAPITAL = 2.0  # Starting capital in USD
DAILY_PROFIT_TARGET = 10.0  # Daily profit target in USD
MAX_COINS = 3  # Number of coins to trade simultaneously

# Trading parameters
TRADE_SIZE_PERCENTAGE = 0.1  # Percentage of available capital per trade
MAX_LEVERAGE = 5  # Maximum leverage to use
STOP_LOSS_PERCENTAGE = 0.5  # Stop loss percentage
TAKE_PROFIT_PERCENTAGE = 1.0  # Take profit percentage

# Coin selection parameters
MIN_24H_VOLUME = 1000000  # Minimum 24h volume in USD
MIN_VOLATILITY = 0.02  # Minimum 24h price change percentage
MAX_SPREAD = 0.001  # Maximum bid-ask spread

# Timeframes
ANALYSIS_TIMEFRAME = '5m'  # Timeframe for analysis
SCALPING_TIMEFRAME = '1m'  # Timeframe for scalping

# Technical indicators
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# API settings
BYBIT_TESTNET = False  # Use mainnet for live trading
API_RATE_LIMIT = 10  # Maximum API calls per second

# Logging
LOG_LEVEL = 'INFO'
SAVE_TRADES = True
