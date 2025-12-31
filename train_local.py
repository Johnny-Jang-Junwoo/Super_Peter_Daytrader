"""
Local AI Training Pipeline

This script processes downloaded CSV files and trains the behavioral cloning model
on your local machine (where RAM is not limited like Azure Free Tier).
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot import (
    DataLoader,
    FeatureEngineer,
    FeatureEngineerConfig,
    BehavioralCloner,
    BehavioralClonerConfig,
)


# Configuration
INBOX_DIR = "data_pipeline/incoming"
MODEL_OUTPUT_DIR = "models"
PROCESSED_DIR = "data_pipeline/processed"


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [INBOX_DIR, MODEL_OUTPUT_DIR, PROCESSED_DIR]:
        os.makedirs(directory, exist_ok=True)


def get_csv_files():
    """Get all CSV files from the inbox."""
    if not os.path.exists(INBOX_DIR):
        return []

    csv_files = [f for f in os.listdir(INBOX_DIR) if f.endswith('.csv')]
    return [os.path.join(INBOX_DIR, f) for f in csv_files]


def train_model():
    """Main training pipeline."""

    print("=" * 80)
    print("🤖 SUPER PETER LOCAL AI TRAINER")
    print("=" * 80)

    ensure_directories()

    # Step 1: Find CSV files
    csv_files = get_csv_files()

    if not csv_files:
        print(f"\n❌ No CSV files found in '{INBOX_DIR}'")
        print("\n💡 Did you forget to run fetch_data.py first?")
        print("   Run: python fetch_data.py")
        return

    print(f"\n📁 Found {len(csv_files)} CSV file(s) to process:")
    for f in csv_files:
        print(f"   • {os.path.basename(f)}")

    # Step 2: Load trade data
    print("\n" + "=" * 80)
    print("STEP 1: Loading Trade Data")
    print("=" * 80)

    loader = DataLoader()
    all_trades = []

    for csv_file in csv_files:
        print(f"\n📄 Processing: {os.path.basename(csv_file)}")
        try:
            trades = loader.load_trades(csv_file)
            all_trades.append(trades)
            print(f"   ✓ Loaded {len(trades)} trades")
        except Exception as e:
            print(f"   ❌ Error loading {csv_file}: {e}")
            continue

    if not all_trades:
        print("\n❌ Failed to load any trade data!")
        return

    # Combine all trades
    import pandas as pd
    combined_trades = pd.concat(all_trades, ignore_index=True)
    print(f"\n✅ Total trades combined: {len(combined_trades)}")

    # Step 3: Fetch market data
    print("\n" + "=" * 80)
    print("STEP 2: Fetching Market Data")
    print("=" * 80)

    symbol = combined_trades["symbol"].iloc[0]
    start_date = combined_trades["timestamp"].min()
    end_date = combined_trades["timestamp"].max()

    print(f"\n📊 Symbol: {symbol}")
    print(f"📅 Date range: {start_date} to {end_date}")

    try:
        market_df = loader.fetch_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1m"
        )

        if market_df.empty:
            print("\n⚠️  No market data available from yfinance")
            print("   Note: 1-minute data only available for last 7-30 days")
            print("   Creating synthetic data for demonstration...")

            # Create synthetic data
            import numpy as np
            timestamps = pd.date_range(start=start_date, end=end_date, freq="1min")
            base_price = 21450.0
            num_candles = len(timestamps)
            np.random.seed(42)

            price_changes = np.random.randn(num_candles) * 5
            close_prices = base_price + np.cumsum(price_changes)

            market_df = pd.DataFrame({
                "timestamp": timestamps,
                "open": close_prices + np.random.randn(num_candles) * 2,
                "high": close_prices + np.abs(np.random.randn(num_candles)) * 3,
                "low": close_prices - np.abs(np.random.randn(num_candles)) * 3,
                "close": close_prices,
                "volume": np.random.randint(100, 1000, num_candles),
                "symbol": symbol,
            })
            market_df["high"] = market_df[["high", "close"]].max(axis=1)
            market_df["low"] = market_df[["low", "close"]].min(axis=1)

        print(f"✅ Retrieved {len(market_df)} market candles")

    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return

    # Step 4: Create training set
    print("\n" + "=" * 80)
    print("STEP 3: Creating Training Set")
    print("=" * 80)

    training_set = loader.create_training_set(
        combined_trades,
        market_df,
        verbose=True
    )

    # Step 5: Add features
    print("\n" + "=" * 80)
    print("STEP 4: Feature Engineering")
    print("=" * 80)

    feature_config = FeatureEngineerConfig(
        rsi_length=14,
        ema_length=20,
    )
    feature_engineer = FeatureEngineer(feature_config)

    training_set = training_set.rename(columns={"timestamp": "date"})
    training_set = feature_engineer.transform(training_set)

    # Add custom features
    training_set["price_change"] = training_set["close"].pct_change()
    training_set["volume_change"] = training_set["volume"].pct_change()
    training_set["high_low_spread"] = (training_set["high"] - training_set["low"]) / training_set["close"]

    # Drop NaN rows
    training_set = training_set.dropna().reset_index(drop=True)

    print(f"\n✅ Training set ready: {len(training_set)} samples")
    print(f"📊 Features: {['close', 'volume', 'rsi', 'ema', 'price_change', 'volume_change', 'high_low_spread']}")

    # Step 6: Train model
    print("\n" + "=" * 80)
    print("STEP 5: Training AI Model")
    print("=" * 80)

    cloner_config = BehavioralClonerConfig(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        test_size=0.2,
        class_weight="balanced"
    )
    cloner = BehavioralCloner(cloner_config)

    feature_columns = [
        "close", "volume", "rsi", "ema",
        "price_change", "volume_change", "high_low_spread"
    ]

    X = training_set[feature_columns]
    y = training_set["target"]

    print(f"\nTraining with {len(X)} samples...")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    try:
        metrics = cloner.train_model(X, y, verbose=True)
        print(f"\n✅ Training complete!")
        print(f"   Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"   Test Accuracy: {metrics['test_accuracy']:.2%}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return

    # Step 7: Save model
    print("\n" + "=" * 80)
    print("STEP 6: Saving Model")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"behavioral_cloner_{symbol}_{timestamp}.pkl"
    model_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)

    cloner.save_brain(model_path)
    print(f"✅ Model saved to: {model_path}")

    # Step 8: Move processed files
    print("\n" + "=" * 80)
    print("STEP 7: Archiving Processed Files")
    print("=" * 80)

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        destination = os.path.join(PROCESSED_DIR, filename)

        try:
            import shutil
            shutil.move(csv_file, destination)
            print(f"📦 Archived: {filename}")
        except Exception as e:
            print(f"⚠️  Could not archive {filename}: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✅ Processed {len(csv_files)} CSV file(s)")
    print(f"✅ Trained on {len(training_set)} samples")
    print(f"✅ Model saved: {model_filename}")
    print(f"✅ Files archived to: {PROCESSED_DIR}")
    print("\n💡 Next steps:")
    print("   • Use the trained model for predictions")
    print("   • Upload more CSV files to continue training")
    print("   • Check 'models/' folder for saved models")
    print("=" * 80)


if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
