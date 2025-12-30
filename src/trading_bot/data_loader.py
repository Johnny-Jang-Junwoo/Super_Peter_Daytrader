from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass
class DataLoaderConfig:
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1d"
    tz: str = "UTC"


class DataLoader:
    """
    DataLoader handles fetching and processing market data and trade orders.

    Supports:
    - Historical OHLCV data from yfinance
    - Trade order CSV parsing and cleaning
    - Merging trades with market data for training
    """

    # Symbol mapping for futures contracts
    # Maps broker symbols to Yahoo Finance tickers
    SYMBOL_MAP = {
        "MNQ": "MNQ=F",  # Micro E-mini Nasdaq-100 Futures
        "NQ": "NQ=F",    # E-mini Nasdaq-100 Futures
        "MES": "MES=F",  # Micro E-mini S&P 500 Futures
        "ES": "ES=F",    # E-mini S&P 500 Futures
        "MYM": "MYM=F",  # Micro E-mini Dow Futures
        "YM": "YM=F",    # E-mini Dow Futures
        "M2K": "M2K=F",  # Micro E-mini Russell 2000 Futures
        "RTY": "RTY=F",  # E-mini Russell 2000 Futures
        "GC": "GC=F",    # Gold Futures
        "SI": "SI=F",    # Silver Futures
        "CL": "CL=F",    # Crude Oil Futures
    }

    def __init__(self, config: Optional[DataLoaderConfig] = None) -> None:
        self.config = config or DataLoaderConfig()

    def fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        data = yf.download(
            ticker,
            start=self.config.start,
            end=self.config.end,
            interval=self.config.interval,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            return data

        data = data.reset_index()

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        # Normalize column names
        data.columns = [str(c).lower().replace(" ", "_") for c in data.columns]

        if "date" not in data.columns and "datetime" in data.columns:
            data = data.rename(columns={"datetime": "date"})

        data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_convert(self.config.tz).dt.normalize()
        data["symbol"] = ticker
        return data

    def fetch_news(self, ticker: str) -> pd.DataFrame:
        # Placeholder for a real news/headline ingestion pipeline.
        return pd.DataFrame(columns=["date", "headline", "sentiment_score"])

    def load(self, ticker: str) -> pd.DataFrame:
        ohlcv = self.fetch_ohlcv(ticker)
        if ohlcv.empty:
            return ohlcv

        news = self.fetch_news(ticker)
        if news.empty:
            ohlcv["sentiment_score"] = 0.5
            return ohlcv

        news = news.copy()
        news["date"] = pd.to_datetime(news["date"], utc=True).dt.tz_convert(self.config.tz).dt.normalize()
        news_daily = (
            news.groupby("date", as_index=False)["sentiment_score"]
            .mean()
            .astype({"sentiment_score": float})
        )
        merged = pd.merge(ohlcv, news_daily, on="date", how="left")
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.5)
        return merged

    def load_trades(self, file_path: str | Path) -> pd.DataFrame:
        """
        Load and clean trade order data from a CSV file.

        Args:
            file_path: Path to the Orders.csv file

        Returns:
            Clean DataFrame with columns: ['timestamp', 'symbol', 'execution_price', 'target']

        Expected CSV columns:
            - Fill Time: Timestamp of order execution (e.g., "12/30/2025 11:27:29")
            - Product: Trading symbol (e.g., "MNQ")
            - B/S: Buy or Sell indicator (e.g., " Buy", " Sell")
            - Status: Order status (e.g., " Filled")
            - Exec Price: Execution price (optional, if available)
        """
        # Read the CSV file
        df = pd.read_csv(file_path)

        print(f"\n{'='*60}")
        print("LOADING TRADE ORDERS")
        print(f"{'='*60}")
        print(f"Total orders in file: {len(df)}")

        # Filter for filled orders only (handle whitespace)
        df["Status"] = df["Status"].str.strip()
        df = df[df["Status"].str.contains("Filled", case=False, na=False)]
        print(f"Filled orders: {len(df)}")

        if df.empty:
            print("WARNING: No filled orders found!")
            return pd.DataFrame(columns=["timestamp", "symbol", "execution_price", "target"])

        # Parse Fill Time column to datetime
        # Try common datetime formats
        try:
            df["timestamp"] = pd.to_datetime(df["Fill Time"])
        except Exception as e:
            # Try alternate column names if "Fill Time" doesn't exist
            possible_time_columns = ["Time", "Timestamp", "DateTime", "Fill_Time"]
            time_col = None
            for col in possible_time_columns:
                if col in df.columns:
                    time_col = col
                    break

            if time_col:
                df["timestamp"] = pd.to_datetime(df[time_col])
            else:
                raise ValueError(
                    f"Could not find time column. Available columns: {df.columns.tolist()}"
                )

        # Floor timestamps to nearest minute for matching with 1-minute OHLCV data
        df["timestamp"] = df["timestamp"].dt.floor("1min")

        # Localize to UTC if not already timezone-aware
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # Extract symbol (handle whitespace)
        # Try "Product" column first, then alternatives
        if "Product" in df.columns:
            df["symbol"] = df["Product"].str.strip()
        elif "Symbol" in df.columns:
            df["symbol"] = df["Symbol"].str.strip()
        elif "Instrument" in df.columns:
            df["symbol"] = df["Instrument"].str.strip()
        else:
            raise ValueError(f"Could not find symbol column. Available columns: {df.columns.tolist()}")

        # Map B/S column to target (1 for Buy, -1 for Sell)
        df["B/S"] = df["B/S"].str.strip()

        def map_action(action: str) -> int:
            action_lower = action.lower()
            if action_lower in ["buy", "b", "long"]:
                return 1
            elif action_lower in ["sell", "s", "short"]:
                return -1
            else:
                print(f"WARNING: Unknown action '{action}', treating as Hold")
                return 0

        df["target"] = df["B/S"].apply(map_action)

        # Extract execution price if available
        price_columns = ["Exec Price", "Price", "Fill Price", "Execution Price", "Avg Price"]
        execution_price_col = None

        for col in price_columns:
            if col in df.columns:
                execution_price_col = col
                break

        if execution_price_col:
            df["execution_price"] = pd.to_numeric(df[execution_price_col], errors="coerce")
        else:
            print("WARNING: No execution price column found, setting to NaN")
            df["execution_price"] = float("nan")

        # Select and order final columns
        result = df[["timestamp", "symbol", "execution_price", "target"]].copy()

        # Remove any rows with invalid targets
        result = result[result["target"] != 0]

        # Print summary statistics
        print(f"\nCleaned trades: {len(result)}")
        buy_count = (result["target"] == 1).sum()
        sell_count = (result["target"] == -1).sum()
        print(f"  Buy orders: {buy_count}")
        print(f"  Sell orders: {sell_count}")

        symbols = result["symbol"].unique()
        print(f"  Symbols: {', '.join(symbols)}")

        print(f"  Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
        print(f"{'='*60}\n")

        return result

    def fetch_market_data(
        self,
        symbol: str,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        interval: str = "1m",
    ) -> pd.DataFrame:
        """
        Fetch 1-minute interval market data from yfinance.

        Args:
            symbol: Trading symbol (will be mapped using SYMBOL_MAP if needed)
            start_date: Start date for data fetch
            end_date: End date for data fetch
            interval: Data interval (default: "1m" for 1-minute)

        Returns:
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        # Map symbol if it's a futures contract
        yahoo_symbol = self.SYMBOL_MAP.get(symbol, symbol)

        print(f"\n{'='*60}")
        print("FETCHING MARKET DATA")
        print(f"{'='*60}")
        print(f"Symbol: {symbol} -> {yahoo_symbol}")
        print(f"Interval: {interval}")
        print(f"Start: {start_date}")
        print(f"End: {end_date}")

        # Download data from yfinance
        data = yf.download(
            yahoo_symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            print(f"WARNING: No data retrieved for {yahoo_symbol}")
            print(f"{'='*60}\n")
            return pd.DataFrame()

        # Reset index to make datetime a column
        data = data.reset_index()

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        # Rename Datetime column to timestamp
        if "Datetime" in data.columns:
            data = data.rename(columns={"Datetime": "timestamp"})
        elif "Date" in data.columns:
            data = data.rename(columns={"Date": "timestamp"})

        # Ensure timestamp is timezone-aware (UTC)
        if data["timestamp"].dt.tz is None:
            data["timestamp"] = data["timestamp"].dt.tz_localize("UTC")
        else:
            data["timestamp"] = data["timestamp"].dt.tz_convert("UTC")

        # Standardize column names
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        data = data.rename(columns=column_mapping)

        # Select final columns
        columns_to_keep = ["timestamp", "open", "high", "low", "close", "volume"]
        result = data[columns_to_keep].copy()

        # Add symbol column
        result["symbol"] = symbol

        print(f"Retrieved {len(result)} candles")
        print(f"Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
        print(f"{'='*60}\n")

        return result

    def create_training_set(
        self,
        trades_df: pd.DataFrame,
        market_df: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Merge trade orders with market data to create a labeled training set.

        Args:
            trades_df: DataFrame from load_trades() with columns ['timestamp', 'symbol', 'execution_price', 'target']
            market_df: DataFrame from fetch_market_data() with OHLCV data
            verbose: Whether to print detailed statistics

        Returns:
            DataFrame with market data and target column (1=Buy, -1=Sell, 0=Hold)
        """
        if verbose:
            print(f"\n{'='*60}")
            print("CREATING TRAINING SET")
            print(f"{'='*60}")

        # Ensure both DataFrames have timestamp columns
        if "timestamp" not in trades_df.columns or "timestamp" not in market_df.columns:
            raise ValueError("Both DataFrames must have 'timestamp' column")

        # Create a copy of market data
        result = market_df.copy()

        # Merge trades into market data on timestamp
        # Use left join to keep all market data rows
        trades_subset = trades_df[["timestamp", "target"]].copy()

        # If there are multiple trades at the same timestamp, keep the first one
        trades_subset = trades_subset.drop_duplicates(subset=["timestamp"], keep="first")

        # Merge
        result = result.merge(trades_subset, on="timestamp", how="left")

        # Fill NaN targets with 0 (Hold)
        result["target"] = result["target"].fillna(0).astype(int)

        # Calculate matching statistics
        total_trades = len(trades_df)
        buy_trades = (trades_df["target"] == 1).sum()
        sell_trades = (trades_df["target"] == -1).sum()

        matched_buys = (result["target"] == 1).sum()
        matched_sells = (result["target"] == -1).sum()
        holds = (result["target"] == 0).sum()

        if verbose:
            print(f"Total market candles: {len(result)}")
            print(f"\nTrade Matching Results:")
            print(f"  Original trades: {total_trades} ({buy_trades} buys, {sell_trades} sells)")
            print(f"  Matched buys: {matched_buys}/{buy_trades} ({100*matched_buys/buy_trades:.1f}%)")
            print(f"  Matched sells: {matched_sells}/{sell_trades} ({100*matched_sells/sell_trades:.1f}%)")
            print(f"  Hold candles: {holds}")

            # Calculate match rate
            total_matched = matched_buys + matched_sells
            match_rate = 100 * total_matched / total_trades if total_trades > 0 else 0
            print(f"\nOverall match rate: {total_matched}/{total_trades} ({match_rate:.1f}%)")

            # Warn if match rate is low
            if match_rate < 80:
                print(f"\n{'!'*60}")
                print("WARNING: Low match rate!")
                print(f"{'!'*60}")
                print("Possible causes:")
                print("  1. Timestamp mismatch (check timezone and rounding)")
                print("  2. Market data doesn't cover the same time period as trades")
                print("  3. Different symbols between trades and market data")
                print("\nRecommendations:")
                print("  - Verify timestamp formats match")
                print("  - Ensure market data covers trade dates")
                print("  - Check symbol mapping (futures contracts)")
                print(f"{'!'*60}")

            print(f"\nTarget distribution:")
            print(f"  Sell (-1): {matched_sells:5d} ({100*matched_sells/len(result):.2f}%)")
            print(f"  Hold (0):  {holds:5d} ({100*holds/len(result):.2f}%)")
            print(f"  Buy (1):   {matched_buys:5d} ({100*matched_buys/len(result):.2f}%)")
            print(f"{'='*60}\n")

        return result
