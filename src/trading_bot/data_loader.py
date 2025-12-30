from __future__ import annotations

from dataclasses import dataclass
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
        data.columns = [c.lower().replace(" ", "_") for c in data.columns]
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
