# Data Pipeline Integration Guide

## Overview

The Data Pipeline provides a complete workflow for integrating trade order data with market data to create labeled training sets for machine learning models.

## Key Features

- **Trade Order Loading**: Parse and clean CSV files from trading platforms
- **Market Data Fetching**: Download 1-minute OHLCV data from yfinance
- **Symbol Mapping**: Automatic translation for futures contracts (MNQ → MNQ=F)
- **Timestamp Alignment**: Floor trade times to nearest minute for matching
- **Data Merging**: Intelligent merge with match rate verification
- **Whitespace Handling**: Robust parsing of messy CSV files

## Quick Start

```python
from trading_bot import DataLoader

# Initialize
loader = DataLoader()

# Load trade orders
trades = loader.load_trades("Orders.csv")

# Fetch market data
market_data = loader.fetch_market_data(
    symbol="MNQ",
    start_date="2024-12-30 09:00",
    end_date="2024-12-30 15:00",
    interval="1m"
)

# Create training set
training_set = loader.create_training_set(trades, market_data)
```

## CSV File Format

### Expected Columns

Your `Orders.csv` should have these columns:

| Column | Description | Example | Required |
|--------|-------------|---------|----------|
| Fill Time | Execution timestamp | "12/30/2025 11:27:29" | Yes |
| Product | Trading symbol | "MNQ" | Yes |
| B/S | Buy or Sell | " Buy" or " Sell" | Yes |
| Status | Order status | " Filled" | Yes |
| Exec Price | Execution price | 21450.25 | Optional |
| Qty | Quantity | 1 | Optional |

### Example CSV

```csv
Fill Time,Product,B/S,Status,Exec Price,Qty
12/30/2024 09:31:15,MNQ, Buy, Filled,21450.25,1
12/30/2024 09:35:42,MNQ, Sell, Filled,21455.50,1
12/30/2024 09:42:18,MNQ, Buy, Filled,21448.75,1
```

### Notes on CSV Format

- **Whitespace**: Leading/trailing spaces are automatically stripped
- **Status Filtering**: Only "Filled" orders are kept
- **Date Format**: Supports common US date formats (MM/DD/YYYY)
- **Action Mapping**: Accepts "Buy", "Sell", "B", "S", "Long", "Short"

## Method Reference

### `load_trades(file_path: str | Path) -> pd.DataFrame`

Load and clean trade orders from a CSV file.

**Parameters:**
- `file_path`: Path to the Orders.csv file

**Returns:**
- DataFrame with columns: `['timestamp', 'symbol', 'execution_price', 'target']`
  - `timestamp`: Timezone-aware datetime (UTC), floored to nearest minute
  - `symbol`: Trading symbol (e.g., "MNQ")
  - `execution_price`: Fill price (float, may be NaN)
  - `target`: 1 for Buy, -1 for Sell

**Example:**
```python
trades_df = loader.load_trades("data/Orders.csv")

# Output:
# ============================================================
# LOADING TRADE ORDERS
# ============================================================
# Total orders in file: 30
# Filled orders: 28
#
# Cleaned trades: 28
#   Buy orders: 14
#   Sell orders: 14
#   Symbols: MNQ
#   Date range: 2024-12-30 09:31:00+00:00 to 2024-12-30 15:15:00+00:00
# ============================================================
```

### `fetch_market_data(symbol, start_date, end_date, interval="1m") -> pd.DataFrame`

Fetch OHLCV market data from yfinance.

**Parameters:**
- `symbol`: Trading symbol (will be mapped using SYMBOL_MAP)
- `start_date`: Start date (string or pd.Timestamp)
- `end_date`: End date (string or pd.Timestamp)
- `interval`: Data interval (default: "1m")

**Returns:**
- DataFrame with columns: `['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']`

**Example:**
```python
market_df = loader.fetch_market_data(
    symbol="MNQ",
    start_date="2024-12-30 09:00",
    end_date="2024-12-30 15:00",
    interval="1m"
)

# Output:
# ============================================================
# FETCHING MARKET DATA
# ============================================================
# Symbol: MNQ -> MNQ=F
# Interval: 1m
# Start: 2024-12-30 09:00:00
# End: 2024-12-30 15:00:00
# Retrieved 360 candles
# Date range: 2024-12-30 09:00:00+00:00 to 2024-12-30 15:00:00+00:00
# ============================================================
```

**Important Notes:**
- yfinance only provides 1-minute data for the **last 7-30 days**
- For historical data, use daily interval: `interval="1d"`
- Timestamps are automatically converted to UTC

### `create_training_set(trades_df, market_df, verbose=True) -> pd.DataFrame`

Merge trade orders with market data to create a labeled training set.

**Parameters:**
- `trades_df`: DataFrame from `load_trades()`
- `market_df`: DataFrame from `fetch_market_data()`
- `verbose`: Print detailed statistics (default: True)

**Returns:**
- DataFrame with market data + `target` column (1=Buy, -1=Sell, 0=Hold)

**Example:**
```python
training_set = loader.create_training_set(trades_df, market_df)

# Output:
# ============================================================
# CREATING TRAINING SET
# ============================================================
# Total market candles: 345
#
# Trade Matching Results:
#   Original trades: 28 (14 buys, 14 sells)
#   Matched buys: 14/14 (100.0%)
#   Matched sells: 14/14 (100.0%)
#   Hold candles: 317
#
# Overall match rate: 28/28 (100.0%)
#
# Target distribution:
#   Sell (-1):    14 (4.06%)
#   Hold (0):    317 (91.88%)
#   Buy (1):      14 (4.06%)
# ============================================================
```

**Low Match Rate Warning:**

If match rate is < 80%, you'll see:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Low match rate!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Possible causes:
  1. Timestamp mismatch (check timezone and rounding)
  2. Market data doesn't cover the same time period as trades
  3. Different symbols between trades and market data
```

## Symbol Mapping

The DataLoader includes built-in mapping for futures contracts:

```python
SYMBOL_MAP = {
    "MNQ": "MNQ=F",  # Micro E-mini Nasdaq-100
    "NQ": "NQ=F",    # E-mini Nasdaq-100
    "MES": "MES=F",  # Micro E-mini S&P 500
    "ES": "ES=F",    # E-mini S&P 500
    "MYM": "MYM=F",  # Micro E-mini Dow
    "YM": "YM=F",    # E-mini Dow
    "M2K": "M2K=F",  # Micro E-mini Russell 2000
    "RTY": "RTY=F",  # E-mini Russell 2000
    "GC": "GC=F",    # Gold Futures
    "SI": "SI=F",    # Silver Futures
    "CL": "CL=F",    # Crude Oil Futures
}
```

**Adding Custom Mappings:**

```python
# Add your own symbol mappings
DataLoader.SYMBOL_MAP["CUSTOM"] = "CUSTOM=F"
```

## Complete Workflow Example

```python
from trading_bot import DataLoader, FeatureEngineer, BehavioralCloner

# Step 1: Load trade orders
loader = DataLoader()
trades = loader.load_trades("Orders.csv")

# Step 2: Fetch market data
symbol = trades["symbol"].iloc[0]
start_date = trades["timestamp"].min()
end_date = trades["timestamp"].max()

market_data = loader.fetch_market_data(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    interval="1m"
)

# Step 3: Create training set
training_set = loader.create_training_set(trades, market_data)

# Step 4: Add technical indicators
feature_engineer = FeatureEngineer()
training_set = training_set.rename(columns={"timestamp": "date"})
training_set = feature_engineer.transform(training_set)

# Step 5: Train AI model
cloner = BehavioralCloner()
X = training_set[["close", "rsi", "ema", "volume"]]
y = training_set["target"]
cloner.train_model(X, y)

# Step 6: Save model
cloner.save_brain("models/my_model.pkl")
```

## Common Issues and Solutions

### Issue 1: "No market data retrieved"

**Cause:** yfinance limitations on 1-minute data

**Solution:**
```python
# Option 1: Use recent dates (within last 7 days)
market_data = loader.fetch_market_data(
    symbol="MNQ",
    start_date="2025-01-05 09:00",  # Recent date
    end_date="2025-01-05 15:00",
    interval="1m"
)

# Option 2: Use daily interval for historical data
market_data = loader.fetch_market_data(
    symbol="MNQ",
    start_date="2024-01-01",
    end_date="2024-12-31",
    interval="1d"  # Daily instead of 1-minute
)
```

### Issue 2: Low match rate

**Causes:**
- Timestamp mismatch
- Timezone issues
- Symbol mismatch

**Solutions:**
```python
# Check timestamps are aligned
print(trades["timestamp"].head())
print(market_data["timestamp"].head())

# Verify symbols match
print(f"Trade symbol: {trades['symbol'].iloc[0]}")
print(f"Market symbol: {market_data['symbol'].iloc[0]}")

# Check timezones
print(f"Trades TZ: {trades['timestamp'].dt.tz}")
print(f"Market TZ: {market_data['timestamp'].dt.tz}")
```

### Issue 3: "Could not find time column"

**Cause:** CSV has different column names

**Solution:**

The loader tries these column names automatically:
- "Fill Time" (default)
- "Time"
- "Timestamp"
- "DateTime"
- "Fill_Time"

If your CSV uses a different name, rename it:
```python
import pandas as pd
df = pd.read_csv("Orders.csv")
df = df.rename(columns={"Your_Time_Column": "Fill Time"})
df.to_csv("Orders_fixed.csv", index=False)
```

### Issue 4: Whitespace in CSV

**Cause:** Trading platforms often export with extra spaces

**Solution:**

The loader automatically handles this! It strips whitespace from:
- Status column: " Filled" → "Filled"
- B/S column: " Buy" → "Buy"
- Product column: " MNQ " → "MNQ"

No action needed on your part.

## Timestamp Alignment

### Why Floor to Minutes?

Trade orders execute at precise millisecond timestamps, but 1-minute OHLCV candles are aligned to minute boundaries. The loader floors trade timestamps to match:

```
Trade execution: 2024-12-30 11:27:29.432
Floored to:      2024-12-30 11:27:00.000
Market candle:   2024-12-30 11:27:00.000 ✓ Match!
```

### Custom Alignment

For different intervals:

```python
# For 5-minute candles
df["timestamp"] = df["timestamp"].dt.floor("5min")

# For hourly candles
df["timestamp"] = df["timestamp"].dt.floor("1h")

# For daily candles
df["timestamp"] = df["timestamp"].dt.normalize()
```

## Performance Tips

### 1. Batch Processing

Process multiple symbols efficiently:

```python
symbols = ["MNQ", "ES", "GC"]
all_data = []

for symbol in symbols:
    trades = loader.load_trades(f"Orders_{symbol}.csv")
    market = loader.fetch_market_data(symbol, start, end)
    training = loader.create_training_set(trades, market)
    all_data.append(training)

combined = pd.concat(all_data, ignore_index=True)
```

### 2. Caching Market Data

Save market data to avoid repeated downloads:

```python
# Download once
market_data = loader.fetch_market_data("MNQ", start, end)
market_data.to_csv("market_cache.csv", index=False)

# Reuse
market_data = pd.read_csv("market_cache.csv")
market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
```

### 3. Memory Optimization

For large datasets:

```python
# Use smaller dtypes
training_set["target"] = training_set["target"].astype("int8")
training_set["volume"] = training_set["volume"].astype("int32")

# Drop unnecessary columns
training_set = training_set.drop(columns=["symbol", "execution_price"])
```

## Data Quality Checks

### Verify Data Integrity

```python
# Check for gaps in market data
time_diffs = market_data["timestamp"].diff()
gaps = time_diffs[time_diffs > pd.Timedelta("2min")]
if not gaps.empty:
    print(f"Found {len(gaps)} gaps in market data")

# Check for duplicate timestamps
duplicates = market_data[market_data.duplicated("timestamp")]
if not duplicates.empty:
    print(f"Found {len(duplicates)} duplicate timestamps")

# Check trade distribution
print(f"Buy/Sell ratio: {(trades['target']==1).sum()}/{(trades['target']==-1).sum()}")
```

### Data Validation

```python
# Verify OHLC relationships
assert (market_data["high"] >= market_data["close"]).all()
assert (market_data["low"] <= market_data["close"]).all()
assert (market_data["high"] >= market_data["low"]).all()

# Check for missing values
assert not training_set["target"].isna().any()
assert not training_set["close"].isna().any()
```

## Integration with Existing Systems

### From Broker API

Many brokers provide APIs that return trade data:

```python
# Example: Convert API response to DataFrame
api_trades = broker.get_trades(account_id)

trades_df = pd.DataFrame([{
    "timestamp": trade.fill_time,
    "symbol": trade.instrument,
    "execution_price": trade.fill_price,
    "target": 1 if trade.side == "BUY" else -1
} for trade in api_trades])

# Floor timestamps
trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"]).dt.floor("1min")
```

### Export to Other Formats

```python
# Save as Parquet (more efficient)
training_set.to_parquet("training_data.parquet")

# Save as HDF5
training_set.to_hdf("training_data.h5", key="data")

# Save as NumPy arrays
np.save("X_train.npy", X.values)
np.save("y_train.npy", y.values)
```

## Next Steps

1. Run the example: `python examples/data_pipeline_example.py`
2. Prepare your Orders.csv file with recent trade data
3. Fetch corresponding market data
4. Create training set and verify match rate
5. Add features and train AI model
6. Backtest and evaluate performance

## References

- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Pandas Datetime](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Trading Order Types](https://www.investopedia.com/terms/o/order.asp)
