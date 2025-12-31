"""
Super Peter Backtester
Simulates the AI trading on historical data to calculate Profit & Loss (P&L).
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path so we can import our bot modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot import DataLoader, FeatureEngineer, BehavioralCloner

# --- CONFIGURATION ---
MODEL_PATH = "models/model_MNQ_20250101.pkl"  # <--- UPDATE THIS filename!
SYMBOL = "MNQ"                                # The symbol to test
START_DATE = "2024-12-20"                     # Start of simulation
END_DATE = "2024-12-30"                       # End of simulation
INITIAL_CAPITAL = 10000.0                     # Starting cash
QUANTITY = 1                                  # Contracts per trade

def run_backtest():
    print("=" * 80)
    print("🤖 SUPER PETER BACKTESTER")
    print("=" * 80)

    # 1. Load the Brain
    print(f"🧠 Loading Model: {MODEL_PATH}...")
    cloner = BehavioralCloner()
    try:
        cloner.load_brain(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   -> Did you update the MODEL_PATH variable at the top of this script?")
        return

    # 2. Fetch Historical Market Data
    print(f"📉 Fetching market data for {SYMBOL} ({START_DATE} to {END_DATE})...")
    loader = DataLoader()
    market_df = loader.fetch_market_data(SYMBOL, START_DATE, END_DATE, interval="1m")
    
    if market_df.empty:
        print("❌ No market data found. Check your dates/internet.")
        return

    # 3. Prepare Features (The "Eyes" of the AI)
    # We must do exactly what we did during training
    fe = FeatureEngineer()
    # Rename 'timestamp' to 'date' because FeatureEngineer expects 'date' or just works on DataFrame
    # Actually FeatureEngineer works on 'close', it doesn't strictly need 'date' unless for sentiment
    features_df = fe.transform(market_df)
    features_df = features_df.dropna() # Drop first few rows (empty EMA/RSI)

    # 4. Ask AI for Decisions
    print("🔮 Generating AI predictions...")
    
    # Select the columns the model expects
    # (We get this list from the loaded brain automatically)
    X = features_df[cloner.feature_columns]
    
    # Predict: 1 (Buy), -1 (Sell), 0 (Hold)
    features_df['ai_signal'] = cloner.predict(X)

    # 5. Run the Simulation (Vectorized P&L)
    # Strategy: If Signal is 1, we are LONG. If -1, we are SHORT. If 0, we are FLAT.
    # Note: This is a simple "Always in the market" or "Signal based" simulation.
    
    print("💰 Calculating profits...")
    
    # Calculate price change from previous minute
    features_df['price_change'] = features_df['close'].diff()
    
    # Shift signal by 1 because we trade AFTER seeing the candle, realizing profit on the NEXT candle
    features_df['position'] = features_df['ai_signal'].shift(1).fillna(0)
    
    # P&L = Position * Price Change * Quantity
    features_df['pnl'] = features_df['position'] * features_df['price_change'] * QUANTITY
    
    # Cumulative P&L
    features_df['cumulative_pnl'] = features_df['pnl'].cumsum()
    features_df['equity_curve'] = INITIAL_CAPITAL + features_df['cumulative_pnl']

    # 6. Report Results
    total_profit = features_df['pnl'].sum()
    total_trades = (features_df['position'].diff() != 0).sum() # Number of times position changed
    final_equity = features_df['equity_curve'].iloc[-1]
    
    print("\n" + "=" * 80)
    print("📊 BACKTEST RESULTS")
    print("=" * 80)
    print(f"Start Capital:   ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Equity:    ${final_equity:,.2f}")
    print(f"Total Profit:    ${total_profit:,.2f} ({total_profit/INITIAL_CAPITAL:.2%})")
    print(f"Total Actions:   {total_trades}")
    
    # Simple win/loss logic (approximate)
    winning_minutes = len(features_df[features_df['pnl'] > 0])
    losing_minutes = len(features_df[features_df['pnl'] < 0])
    print(f"Winning Minutes: {winning_minutes}")
    print(f"Losing Minutes:  {losing_minutes}")

    print("\n📝 Sample of last 5 decisions:")
    print(features_df[['timestamp', 'close', 'ai_signal', 'cumulative_pnl']].tail())

    # Optional: Plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(features_df['timestamp'], features_df['equity_curve'], label='AI Equity')
        plt.title(f"AI Strategy Performance: {SYMBOL}")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True)
        plt.show()
    except:
        print("\n(Graph could not be shown - matplotlib might be missing)")

if __name__ == "__main__":
    run_backtest()