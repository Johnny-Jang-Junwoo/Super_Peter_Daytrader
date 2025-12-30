"""
Simple Momentum Trading Strategy Example

This strategy buys when RSI is oversold and price is above EMA,
sells when RSI is overbought.
"""

from __future__ import annotations

import pandas as pd

from trading_bot import DataLoader, DataLoaderConfig, FeatureEngineer, RiskManager, RiskLimits, StrategyBase


class MomentumStrategy(StrategyBase):
    """A simple momentum-based trading strategy using RSI and EMA indicators."""

    def __init__(
        self,
        data_loader: DataLoader,
        risk_manager: RiskManager,
        feature_engineer: FeatureEngineer,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ):
        super().__init__(data_loader, risk_manager, feature_engineer)
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on RSI and EMA."""
        df = data.copy()

        # Initialize signal column
        df["signal"] = 0

        # Buy signal: RSI oversold and price above EMA
        df.loc[(df["rsi"] < self.rsi_oversold) & (df["close"] > df["ema"]), "signal"] = 1

        # Sell signal: RSI overbought
        df.loc[df["rsi"] > self.rsi_overbought, "signal"] = -1

        return df

    def size_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate position sizes (simple fixed percentage for this example)."""
        df = data.copy()
        df["position_size"] = 0.0

        # Use 10% of equity per trade
        df.loc[df["signal"] != 0, "position_size"] = 0.1

        return df


def main():
    """Run the momentum strategy example."""
    # Configure data loader for the last year of data
    config = DataLoaderConfig(
        start="2024-01-01",
        end="2024-12-31",
        interval="1d",
    )
    data_loader = DataLoader(config)

    # Set up risk management (10k starting equity, max 500 daily loss, 2k max drawdown)
    risk_limits = RiskLimits(daily_max_loss=500.0, total_drawdown=2000.0)
    risk_manager = RiskManager(starting_equity=10000.0, limits=risk_limits)

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Create strategy
    strategy = MomentumStrategy(
        data_loader=data_loader,
        risk_manager=risk_manager,
        feature_engineer=feature_engineer,
    )

    # Example: Load and process data for a stock
    ticker = "AAPL"
    print(f"Loading data for {ticker}...")

    # Load OHLCV data
    data = data_loader.load(ticker)

    if data.empty:
        print(f"No data available for {ticker}")
        return

    print(f"Loaded {len(data)} rows of data")

    # Add technical indicators
    data = feature_engineer.transform(data)

    # Generate signals
    data = strategy.generate_signals(data)

    # Size positions
    data = strategy.size_position(data)

    # Display results
    signals = data[data["signal"] != 0][["date", "close", "rsi", "ema", "signal", "position_size"]]

    if not signals.empty:
        print(f"\nGenerated {len(signals)} trading signals:")
        print(signals.to_string(index=False))

        buy_signals = len(signals[signals["signal"] == 1])
        sell_signals = len(signals[signals["signal"] == -1])
        print(f"\nSummary: {buy_signals} buy signals, {sell_signals} sell signals")
    else:
        print("\nNo trading signals generated for this period")

    # Check if we can trade based on risk limits
    if strategy.can_trade():
        print("\nRisk manager: Trading is allowed")
    else:
        print("\nRisk manager: Trading is NOT allowed (risk limits exceeded)")


if __name__ == "__main__":
    main()
