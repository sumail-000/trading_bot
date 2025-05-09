# Crypto Scalping Bot

An autonomous, intelligent, and risk-aware crypto trading bot that starts with $2 in a Bybit trading account and aims to make $10 profit per day using ultra-small trades on 3 volatile coins simultaneously.

## Features

- Multi-coin parallel trading engines
- Intelligent market scanning and coin selection
- Scalping strategy with tight stop-losses
- Smart risk management and capital allocation
- Daily profit capping at $10

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Bybit API credentials:
   ```
   BYBIT_API_KEY=your_api_key
   BYBIT_API_SECRET=your_api_secret
   ```
4. Run the bot: `python main.py`

## Architecture

- `main.py`: Entry point for the bot
- `coin_selector.py`: Scans and selects volatile coins
- `trading_engine.py`: Core trading logic for each coin
- `risk_manager.py`: Handles capital allocation and risk control
- `bybit_api.py`: Interface with Bybit exchange
- `indicators.py`: Technical indicators for trading signals
- `config.py`: Configuration settings

## Disclaimer

This bot is for educational purposes only. Cryptocurrency trading involves significant risk. Use at your own risk.
