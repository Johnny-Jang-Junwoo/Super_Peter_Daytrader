"""
Local AI Training Pipeline (Google Drive Edition)

This script trains the behavioral cloning model using data 
DIRECTLY from your Google Drive folder.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot import (
    DataLoader,
    FeatureEngineer,
    FeatureEngineerConfig,
    BehavioralCloner,
    BehavioralClonerConfig,
)

# --- CONFIGURATION ---
# The path you confirmed in verify_local_data.py
GOOGLE_DRIVE_DIR = r"G:\My Drive\SuperPeterTrader"
MODEL_OUTPUT_DIR = "models"

def train_model():
    print("=" * 80)
    print("🤖 SUPER PETER TRADER - GOOGLE DRIVE TRAINING")
    print("=" * 80)

    # 1. Setup Directories
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    drive_path = Path(GOOGLE_DRIVE_DIR)

    # 2. Find CSV Files in Drive
    if not drive_path.exists():
        print(f"❌ Error: Google Drive path not found: {drive_path}")
        return

    csv_files = sorted(drive_path.glob("*.csv"))
    
    if not csv_files:
        print(f"\n❌ No CSV files found in '{drive_path}'")
        print("   -> Please drop your trade CSVs into that Google Drive folder.")
        return

    print(f"\n📁 Found {len(csv_files)} file(s) in Drive:")
    for f in csv_files:
        print(f"   • {f.name}")

    # 3. Load Trade Data
    print("\n" + "=" * 80)
    print("STEP 1: Loading Trade Data")
    print("=" * 80)

    loader = DataLoader()
    all_trades = []

    for csv_file in csv_files:
        print(f"\n📄 Processing: {csv_file.name}")
        try:
            # Note: We pass the full Path object
            trades = loader.load_trades(csv_file)
            if not trades.empty:
                all_trades.append(trades)
                print(f"   ✓ Loaded {len(trades)} trades")
        except Exception as e:
            print(f"   ❌ Error loading {csv_file.name}: {e}")

    if not all_trades:
        print("\n❌ Failed to load any valid trades.")
        return

    combined_trades = pd.concat(all_trades, ignore_index=True)
    combined_trades = combined_trades.sort_values("timestamp")
    print(f"\n✅ Total combined trades: {len(combined_trades)}")

    # 4. Fetch Market Data (yfinance)
    print("\n" + "=" * 80)
    print("STEP 2: Fetching Market Data")
    print("=" * 80)

    # Get symbol and date range from the loaded trades
    symbol = combined_trades["symbol"].iloc[0] # Assuming single symbol for now
    start_date = combined_trades["timestamp"].min() - pd.Timedelta(days=1)
    end_date = combined_trades["timestamp"].max() + pd.Timedelta(days=1)

    print(f"   Target: {symbol}")
    print(f"   Range:  {start_date.date()} to {end_date.date()}")

    try:
        market_df = loader.fetch_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1m" # Standard for day trading
        )
    except Exception as e:
        print(f"❌ API Error: {e}")
        return

    if market_df.empty:
        print("❌ No market data found. Check your internet or the symbol.")
        return

    # 5. Create Training Set (Merge Trades + Market)
    print("\n" + "=" * 80)
    print("STEP 3: Training Model")
    print("=" * 80)

    training_set = loader.create_training_set(combined_trades, market_df)
    
    # Feature Engineering
    fe = FeatureEngineer(FeatureEngineerConfig(rsi_length=14, ema_length=20))
    training_set = training_set.rename(columns={"timestamp": "date"}) # Fix column name for FE
    training_set = fe.transform(training_set)
    
    # Clean up for training
    training_set = training_set.dropna()
    
    # Prepare X (Features) and y (Target)
    features = ["close", "volume", "rsi", "ema", "sentiment_score"]
    X = training_set[features]
    y = training_set["target"]

    # Train
    cloner = BehavioralCloner(BehavioralClonerConfig(n_estimators=100))
    metrics = cloner.train_model(X, y)

    # 6. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(MODEL_OUTPUT_DIR) / f"model_{symbol}_{timestamp}.pkl"
    cloner.save_brain(save_path)

    print("\n" + "=" * 80)
    print(f"🚀 SUCCESS! Model saved to: {save_path}")
    print("=" * 80)

if __name__ == "__main__":
    train_model()