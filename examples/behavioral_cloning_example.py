"""
Behavioral Cloning Example

This example demonstrates how to:
1. Load historical market data
2. Add technical indicators
3. Load trade logs from an expert trader
4. Train an AI model to mimic the trading decisions
5. Save and load the trained model
6. Make predictions on new data
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from trading_bot import (
    BehavioralCloner,
    BehavioralClonerConfig,
    DataLoader,
    DataLoaderConfig,
    FeatureEngineer,
    FeatureEngineerConfig,
)


def create_sample_trade_log(market_data: pd.DataFrame, output_file: str) -> None:
    """
    Create a sample trade log CSV for demonstration purposes.

    In a real scenario, this would be your friend's actual trade history.
    """
    # Select some dates for sample trades
    dates = market_data["date"].values

    # Create sample trades (this is just for demonstration)
    trades = []

    # Example: Buy when price is low, sell when high (simplified logic)
    prices = market_data["close"].values

    # Add some buy signals at lower prices
    low_price_indices = [10, 25, 45, 78, 120, 145, 189]
    for idx in low_price_indices:
        if idx < len(dates):
            trades.append({"date": dates[idx], "action": "buy"})

    # Add some sell signals at higher prices
    high_price_indices = [20, 35, 60, 95, 130, 160, 200]
    for idx in high_price_indices:
        if idx < len(dates):
            trades.append({"date": dates[idx], "action": "sell"})

    # Create DataFrame and save
    trade_df = pd.DataFrame(trades)
    trade_df.to_csv(output_file, index=False)
    print(f"Sample trade log created: {output_file}")
    print(f"Total trades: {len(trades)}")


def main():
    """Run the behavioral cloning example."""
    print("=" * 80)
    print("BEHAVIORAL CLONING TRADING BOT - EXAMPLE")
    print("=" * 80)

    # Step 1: Load Market Data
    print("\n[Step 1] Loading market data...")
    config = DataLoaderConfig(
        start="2024-01-01",
        end="2024-12-31",
        interval="1d",
    )
    data_loader = DataLoader(config)
    ticker = "AAPL"

    market_data = data_loader.load(ticker)
    if market_data.empty:
        print(f"No data available for {ticker}")
        return

    print(f"Loaded {len(market_data)} rows of market data for {ticker}")

    # Step 2: Add Technical Indicators
    print("\n[Step 2] Adding technical indicators...")
    feature_config = FeatureEngineerConfig(
        rsi_length=14,
        ema_length=20,
    )
    feature_engineer = FeatureEngineer(feature_config)
    market_data = feature_engineer.transform(market_data)

    # Add additional features for better predictions
    market_data["price_change"] = market_data["close"].pct_change()
    market_data["volume_change"] = market_data["volume"].pct_change()
    market_data["high_low_spread"] = (market_data["high"] - market_data["low"]) / market_data["close"]

    # Drop NaN rows created by indicators
    market_data = market_data.dropna().reset_index(drop=True)

    print(f"Features available: {list(market_data.columns)}")

    # Step 3: Create or Load Trade Log
    print("\n[Step 3] Loading trade log...")
    trade_log_file = "data/sample_trades.csv"

    # Create sample trade log if it doesn't exist
    if not Path(trade_log_file).exists():
        Path("data").mkdir(exist_ok=True)
        create_sample_trade_log(market_data, trade_log_file)
    else:
        print(f"Using existing trade log: {trade_log_file}")

    # Step 4: Initialize Behavioral Cloner
    print("\n[Step 4] Initializing Behavioral Cloner...")
    cloner_config = BehavioralClonerConfig(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        test_size=0.2,
        class_weight="balanced",  # Handle imbalanced data
    )
    cloner = BehavioralCloner(cloner_config)

    # Step 5: Prepare Training Data
    print("\n[Step 5] Preparing labeled training data...")
    labeled_data = cloner.prepare_labels(market_data, trade_log_file)

    # Select features for training
    feature_columns = [
        "close",
        "volume",
        "rsi",
        "ema",
        "sentiment_score",
        "price_change",
        "volume_change",
        "high_low_spread",
    ]

    X = labeled_data[feature_columns]
    y = labeled_data["target"]

    print(f"Training data shape: {X.shape}")
    print(f"Features: {feature_columns}")

    # Step 6: Train the Model
    print("\n[Step 6] Training the model...")
    print("This may take a moment...\n")

    metrics = cloner.train_model(X, y, verbose=True)

    # Step 7: Save the Model
    print("[Step 7] Saving the trained model...")
    model_file = "models/behavioral_cloner.pkl"
    cloner.save_brain(model_file)

    # Step 8: Demonstrate Loading and Prediction
    print("\n[Step 8] Loading model and making predictions...")

    # Create a new instance and load the saved model
    new_cloner = BehavioralCloner()
    new_cloner.load_brain(model_file)

    # Make predictions on the last 10 days
    print("\nPredictions for the last 10 trading days:")
    print("-" * 80)

    recent_data = labeled_data.tail(10).copy()
    X_recent = recent_data[feature_columns]
    predictions = new_cloner.predict(X_recent)
    probabilities = new_cloner.predict_proba(X_recent)

    # Map predictions to readable format
    action_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}

    for i, (idx, row) in enumerate(recent_data.iterrows()):
        pred = predictions[i]
        proba = probabilities[i]
        actual = row["target"]

        print(f"Date: {row['date'].strftime('%Y-%m-%d')} | "
              f"Close: ${row['close']:7.2f} | "
              f"RSI: {row['rsi']:5.1f} | "
              f"Predicted: {action_map[pred]:4s} | "
              f"Actual: {action_map[actual]:4s} | "
              f"Confidence: {max(proba)*100:5.1f}%")

    print("-" * 80)

    # Step 9: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model Type: Random Forest Classifier")
    print(f"Training Samples: {metrics['train_size']}")
    print(f"Test Samples: {metrics['test_size']}")
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Model saved to: {model_file}")
    print("=" * 80)

    print("\n[SUCCESS] Behavioral cloning example completed successfully!")
    print("\nNext steps:")
    print("1. Replace 'data/sample_trades.csv' with real trade logs")
    print("2. Experiment with different features and hyperparameters")
    print("3. Integrate the trained model into a live trading strategy")
    print("4. Backtest the model's performance on historical data")


if __name__ == "__main__":
    main()
