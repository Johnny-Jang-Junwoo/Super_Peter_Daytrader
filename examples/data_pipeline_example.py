"""
Data Pipeline Integration Example

This example demonstrates the complete data ingestion workflow:
1. Loading trade orders from CSV
2. Fetching corresponding market data from yfinance
3. Merging trades with market data
4. Creating a labeled training set
5. Training an AI model on the integrated data
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from trading_bot import (
    BehavioralCloner,
    BehavioralClonerConfig,
    DataLoader,
    FeatureEngineer,
    FeatureEngineerConfig,
)


def main():
    """Run the complete data pipeline integration example."""
    print("=" * 80)
    print("DATA PIPELINE INTEGRATION - COMPLETE EXAMPLE")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load Trade Orders from CSV
    # =========================================================================
    print("\n[STEP 1] Loading trade orders from CSV...")

    data_loader = DataLoader()
    orders_file = "data/sample_Orders.csv"

    # Check if file exists
    if not Path(orders_file).exists():
        print(f"ERROR: {orders_file} not found!")
        print("Please ensure the Orders.csv file is in the data/ directory.")
        return

    # Load and clean trade orders
    trades_df = data_loader.load_trades(orders_file)

    if trades_df.empty:
        print("ERROR: No trades loaded. Exiting.")
        return

    # Display sample trades
    print("\nSample trades (first 5):")
    print(trades_df.head().to_string(index=False))

    # =========================================================================
    # STEP 2: Fetch Market Data for the Trade Period
    # =========================================================================
    print("\n[STEP 2] Fetching market data from yfinance...")

    # Get the symbol from trades (assuming single symbol for this example)
    symbol = trades_df["symbol"].iloc[0]
    print(f"Trading symbol: {symbol}")

    # Determine date range from trades
    start_date = trades_df["timestamp"].min()
    end_date = trades_df["timestamp"].max()

    # Add some buffer to ensure we have all data
    start_date = start_date - pd.Timedelta(hours=1)
    end_date = end_date + pd.Timedelta(hours=1)

    print(f"Fetching data from {start_date} to {end_date}")

    # Fetch 1-minute market data
    # NOTE: yfinance has limitations on 1-minute data (typically last 7-30 days)
    market_df = data_loader.fetch_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval="1m",
    )

    if market_df.empty:
        print("\n" + "!" * 80)
        print("WARNING: No market data retrieved!")
        print("!" * 80)
        print("\nPossible reasons:")
        print("1. yfinance only provides recent 1-minute data (last 7-30 days)")
        print("2. The dates in sample_Orders.csv may be too old")
        print("3. Symbol mapping might be incorrect")
        print("\nSolutions:")
        print("1. Use recent dates in Orders.csv (within last 7 days)")
        print("2. For historical data, use daily interval instead of 1-minute")
        print("3. Check DataLoader.SYMBOL_MAP for correct symbol mapping")
        print("\nFor demonstration, creating synthetic market data...")

        # Create synthetic market data for demonstration
        market_df = create_synthetic_market_data(trades_df)

    # Display sample market data
    print("\nSample market data (first 5):")
    print(market_df.head().to_string(index=False))

    # =========================================================================
    # STEP 3: Merge Trades with Market Data
    # =========================================================================
    print("\n[STEP 3] Merging trades with market data...")

    training_set = data_loader.create_training_set(
        trades_df=trades_df,
        market_df=market_df,
        verbose=True,
    )

    # Display sample of labeled data
    print("\nSample labeled training data (first 10):")
    print(training_set[["timestamp", "open", "high", "low", "close", "volume", "target"]].head(10).to_string(index=False))

    # Show some examples of each label
    print("\n--- Examples of BUY signals (target=1) ---")
    buy_samples = training_set[training_set["target"] == 1].head(3)
    if not buy_samples.empty:
        print(buy_samples[["timestamp", "close", "target"]].to_string(index=False))
    else:
        print("No buy signals found")

    print("\n--- Examples of SELL signals (target=-1) ---")
    sell_samples = training_set[training_set["target"] == -1].head(3)
    if not sell_samples.empty:
        print(sell_samples[["timestamp", "close", "target"]].to_string(index=False))
    else:
        print("No sell signals found")

    # =========================================================================
    # STEP 4: Add Technical Indicators
    # =========================================================================
    print("\n[STEP 4] Adding technical indicators...")

    # For 1-minute data, use shorter periods
    feature_config = FeatureEngineerConfig(
        rsi_length=14,
        ema_length=20,
    )
    feature_engineer = FeatureEngineer(feature_config)

    # Rename columns to match FeatureEngineer expectations
    training_set = training_set.rename(columns={"timestamp": "date"})

    # Add indicators
    training_set = feature_engineer.transform(training_set)

    # Add additional features for better predictions
    training_set["price_change"] = training_set["close"].pct_change()
    training_set["volume_change"] = training_set["volume"].pct_change()
    training_set["high_low_spread"] = (training_set["high"] - training_set["low"]) / training_set["close"]

    # Drop NaN rows
    training_set = training_set.dropna().reset_index(drop=True)

    print(f"Training set with features: {len(training_set)} rows")
    print(f"Available features: {list(training_set.columns)}")

    # =========================================================================
    # STEP 5: Train AI Model on Integrated Data
    # =========================================================================
    print("\n[STEP 5] Training AI model on integrated data...")

    # Initialize behavioral cloner
    cloner_config = BehavioralClonerConfig(
        n_estimators=50,  # Reduced for faster training on small dataset
        max_depth=8,
        random_state=42,
        test_size=0.2,
        class_weight="balanced",
    )
    cloner = BehavioralCloner(cloner_config)

    # Select features for training
    feature_columns = [
        "close",
        "volume",
        "rsi",
        "ema",
        "price_change",
        "volume_change",
        "high_low_spread",
    ]

    X = training_set[feature_columns]
    y = training_set["target"]

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:")
    print(y.value_counts().sort_index())

    # Check if we have enough data to train
    if len(training_set) < 10:
        print("\nWARNING: Not enough data to train model (need at least 10 samples)")
        print("Please use a larger Orders.csv file or fetch more market data.")
        return

    # Check if we have any actual trades
    if (y != 0).sum() < 2:
        print("\nWARNING: Not enough trade signals to train model")
        print("Please ensure trades are properly matched with market data.")
        return

    # Train the model
    try:
        metrics = cloner.train_model(X, y, verbose=True)

        # =========================================================================
        # STEP 6: Save the Model
        # =========================================================================
        print("\n[STEP 6] Saving the trained model...")

        model_file = "models/pipeline_trained_model.pkl"
        cloner.save_brain(model_file)

        # =========================================================================
        # STEP 7: Make Predictions
        # =========================================================================
        print("\n[STEP 7] Making predictions on recent data...")

        # Use the last 10 samples for prediction
        recent_data = training_set.tail(10)
        X_recent = recent_data[feature_columns]
        predictions = cloner.predict(X_recent)
        probabilities = cloner.predict_proba(X_recent)

        # Display predictions
        action_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}

        print("\nPredictions for recent data:")
        print("-" * 100)
        print(f"{'Timestamp':<20} {'Close':>10} {'RSI':>6} {'Predicted':>8} {'Actual':>8} {'Confidence':>12}")
        print("-" * 100)

        for i, (idx, row) in enumerate(recent_data.iterrows()):
            pred = predictions[i]
            actual = row["target"]
            confidence = max(probabilities[i]) * 100

            timestamp_str = str(row["date"])[:19]
            print(f"{timestamp_str:<20} {row['close']:>10.2f} {row['rsi']:>6.1f} "
                  f"{action_map[pred]:>8} {action_map[actual]:>8} {confidence:>11.1f}%")

        print("-" * 100)

        # =========================================================================
        # SUMMARY
        # =========================================================================
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        print(f"Trades loaded: {len(trades_df)}")
        print(f"Market candles: {len(market_df)}")
        print(f"Training samples: {len(training_set)}")
        print(f"Model accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Model saved to: {model_file}")
        print("=" * 80)

        print("\n[SUCCESS] Data pipeline integration completed!")
        print("\nNext steps:")
        print("1. Replace sample_Orders.csv with your real trade data")
        print("2. Ensure dates in Orders.csv are recent (within last 7 days for 1m data)")
        print("3. Experiment with different features and model parameters")
        print("4. Backtest the model on out-of-sample data")

    except Exception as e:
        print(f"\nERROR during model training: {e}")
        print("\nThis may be due to insufficient or imbalanced data.")
        print("Try collecting more trade data or adjusting the date range.")


def create_synthetic_market_data(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic market data for demonstration when real data is unavailable.

    Args:
        trades_df: DataFrame with trade information

    Returns:
        Synthetic market data DataFrame
    """
    print("\nCreating synthetic market data for demonstration...")

    # Get time range from trades
    start_time = trades_df["timestamp"].min()
    end_time = trades_df["timestamp"].max()

    # Create 1-minute timestamps
    timestamps = pd.date_range(start=start_time, end=end_time, freq="1min")

    # Generate synthetic OHLCV data
    base_price = 21450.0
    num_candles = len(timestamps)

    # Random walk for prices
    import numpy as np
    np.random.seed(42)

    price_changes = np.random.randn(num_candles) * 5  # Random changes
    close_prices = base_price + np.cumsum(price_changes)

    # Create OHLCV
    market_data = pd.DataFrame({
        "timestamp": timestamps,
        "open": close_prices + np.random.randn(num_candles) * 2,
        "high": close_prices + np.abs(np.random.randn(num_candles)) * 3,
        "low": close_prices - np.abs(np.random.randn(num_candles)) * 3,
        "close": close_prices,
        "volume": np.random.randint(100, 1000, num_candles),
        "symbol": trades_df["symbol"].iloc[0],
    })

    # Ensure high >= close >= low
    market_data["high"] = market_data[["high", "close"]].max(axis=1)
    market_data["low"] = market_data[["low", "close"]].min(axis=1)

    print(f"Created {len(market_data)} synthetic candles")

    return market_data


if __name__ == "__main__":
    main()
