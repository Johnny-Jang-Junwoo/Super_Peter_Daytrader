# Super Peter Daytrader

A Python framework for building and testing algorithmic day trading strategies with built-in risk management and technical analysis capabilities.

## Features

- **Data Loading**: Fetch historical OHLCV data using yfinance
- **Feature Engineering**: Built-in technical indicators (RSI, EMA, etc.)
- **Risk Management**: Configurable daily loss limits and drawdown protection
- **Strategy Framework**: Abstract base class for implementing custom strategies
- **Sentiment Analysis**: Support for news sentiment integration (TextBlob, VADER)
- **AI Behavioral Cloning**: Train machine learning models to mimic expert trading decisions

## Project Structure

```
Super_Peter_Daytrader/
├── src/
│   └── trading_bot/
│       ├── __init__.py           # Package exports
│       ├── data_loader.py        # Data fetching and loading
│       ├── feature_engineer.py   # Technical indicator calculations
│       ├── risk_manager.py       # Risk management and position limits
│       ├── strategy_base.py      # Abstract strategy base class
│       └── ai_trainer.py         # AI Behavioral Cloning (NEW!)
├── examples/
│   ├── momentum_strategy.py      # Example momentum trading strategy
│   └── behavioral_cloning_example.py  # AI training example (NEW!)
├── docs/
│   └── AI_TRAINER_GUIDE.md       # Comprehensive AI Trainer documentation
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Super_Peter_Daytrader
```

2. Create and activate virtual environment (if not already created):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: `pandas-ta` is currently excluded due to Python 3.14 compatibility issues. The framework will use fallback implementations for technical indicators.

## Quick Start

### Running the Example Strategy

```bash
python examples/momentum_strategy.py
```

This will:
- Load AAPL stock data for 2024
- Calculate technical indicators (RSI, EMA)
- Generate buy/sell signals based on momentum
- Display trading signals with position sizes

### Creating Your Own Strategy

1. Inherit from `StrategyBase`:

```python
from trading_bot import StrategyBase
import pandas as pd

class MyStrategy(StrategyBase):
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Implement your signal generation logic."""
        df = data.copy()
        df["signal"] = 0
        # Your logic here: 1 for buy, -1 for sell, 0 for hold
        return df

    def size_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """Implement your position sizing logic."""
        df = data.copy()
        df["position_size"] = 0.0
        # Your position sizing logic here
        return df
```

2. Initialize the components:

```python
from trading_bot import (
    DataLoader, DataLoaderConfig,
    FeatureEngineer, FeatureEngineerConfig,
    RiskManager, RiskLimits
)

# Configure data loader
config = DataLoaderConfig(
    start="2024-01-01",
    end="2024-12-31",
    interval="1d"
)
data_loader = DataLoader(config)

# Set up risk management
risk_limits = RiskLimits(
    daily_max_loss=500.0,
    total_drawdown=2000.0
)
risk_manager = RiskManager(
    starting_equity=10000.0,
    limits=risk_limits
)

# Initialize feature engineer
feature_engineer = FeatureEngineer()

# Create your strategy
strategy = MyStrategy(
    data_loader=data_loader,
    risk_manager=risk_manager,
    feature_engineer=feature_engineer
)
```

3. Load data and generate signals:

```python
# Load market data
data = data_loader.load("AAPL")

# Add technical indicators
data = feature_engineer.transform(data)

# Generate trading signals
data = strategy.generate_signals(data)

# Calculate position sizes
data = strategy.size_position(data)

# Check risk limits
if strategy.can_trade():
    print("Trading allowed")
else:
    print("Risk limits exceeded")
```

## Components

### DataLoader

Fetches market data from Yahoo Finance and processes trade orders:
- **Market Data**: Configurable time periods and intervals
- **OHLCV Data**: Automatic column normalization
- **Trade Orders**: Parse and clean CSV files from trading platforms
- **Symbol Mapping**: Automatic futures contract translation (MNQ → MNQ=F)
- **Data Merging**: Intelligent timestamp alignment and merge with match verification
- **1-Minute Data**: Support for high-frequency trading analysis

#### New: Data Pipeline Integration

```python
# Load trade orders from CSV
trades = loader.load_trades("Orders.csv")

# Fetch 1-minute market data
market_data = loader.fetch_market_data(
    symbol="MNQ",
    start_date="2024-12-30 09:00",
    end_date="2024-12-30 15:00",
    interval="1m"
)

# Create labeled training set
training_set = loader.create_training_set(trades, market_data)
```

**See [docs/DATA_PIPELINE_GUIDE.md](docs/DATA_PIPELINE_GUIDE.md) for complete documentation.**

### FeatureEngineer

Adds technical indicators to your data:
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- Sentiment score integration
- Extensible for custom indicators

### RiskManager

Protects your capital with:
- Daily maximum loss limits
- Total drawdown protection
- Automatic equity tracking
- Daily reset mechanism

### StrategyBase

Abstract base class for strategies requiring:
- `generate_signals()`: Your signal generation logic
- `size_position()`: Your position sizing logic
- Built-in `can_trade()` method for risk checks

### BehavioralCloner (AI Trainer) 🤖

**NEW!** Train AI models to learn from expert trading decisions:
- **Behavioral Cloning**: Learn trading patterns from historical trade logs
- **Random Forest Classifier**: Robust machine learning model
- **Imbalanced Data Handling**: Automatically handles rare trading signals with `class_weight='balanced'`
- **Feature Importance**: Identifies which indicators matter most
- **Model Persistence**: Save and load trained models with joblib

#### Quick Example

```python
from trading_bot import BehavioralCloner, DataLoader, FeatureEngineer

# Load and prepare data
data_loader = DataLoader()
market_data = data_loader.load("AAPL")

feature_engineer = FeatureEngineer()
market_data = feature_engineer.transform(market_data)

# Initialize AI trainer
cloner = BehavioralCloner()

# Prepare labels from your friend's trade log
labeled_data = cloner.prepare_labels(market_data, "trade_logs.csv")

# Train the model
X = labeled_data[["close", "volume", "rsi", "ema"]]
y = labeled_data["target"]
metrics = cloner.train_model(X, y)

# Save the trained model
cloner.save_brain("models/my_strategy.pkl")

# Make predictions
predictions = cloner.predict(X)
```

#### Trade Log Format

Your CSV should have columns: `date`, `action`

```csv
date,action
2024-01-15,buy
2024-01-20,sell
2024-02-03,buy
```

**See [docs/AI_TRAINER_GUIDE.md](docs/AI_TRAINER_GUIDE.md) for comprehensive documentation.**

#### Running the AI Example

```bash
python examples/behavioral_cloning_example.py
```

This demonstrates the complete workflow: loading data, training, evaluation, and making predictions.

## Configuration

### Data Loader Configuration

```python
config = DataLoaderConfig(
    start="2024-01-01",      # Start date (optional)
    end="2024-12-31",        # End date (optional)
    interval="1d",           # Data interval (1m, 5m, 1h, 1d, etc.)
    tz="UTC"                 # Timezone for dates
)
```

### Feature Engineer Configuration

```python
config = FeatureEngineerConfig(
    rsi_length=14,           # RSI period
    ema_length=20,           # EMA period
    sentiment_default=0.5    # Default sentiment score
)
```

### Risk Management Configuration

```python
risk_limits = RiskLimits(
    daily_max_loss=500.0,    # Maximum daily loss in dollars
    total_drawdown=2000.0    # Maximum total drawdown in dollars
)
```

## Development

### Running Tests

```bash
pytest
```

### Code Style

The project follows PEP 8 guidelines and uses type hints throughout.

## Dependencies

- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `yfinance`: Yahoo Finance data fetching
- `scikit-learn`: Machine learning utilities
- `textblob`: Text sentiment analysis
- `vaderSentiment`: Sentiment analysis
- ~~`pandas-ta`: Technical analysis~~ (excluded due to Python 3.14 compatibility)

## Roadmap

- [ ] Backtesting engine with performance metrics
- [ ] Live trading integration (paper trading)
- [ ] Additional technical indicators
- [ ] Machine learning strategy examples
- [ ] Portfolio optimization
- [ ] Multi-asset support
- [ ] Real-time news sentiment integration

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This software is for educational purposes only. Trading stocks and other financial instruments involves risk. Past performance does not guarantee future results. Always do your own research and consider consulting with a qualified financial advisor before making investment decisions.
