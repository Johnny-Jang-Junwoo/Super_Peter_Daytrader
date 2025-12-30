from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional dependency
    ta = None


@dataclass
class FeatureEngineerConfig:
    rsi_length: int = 14
    ema_length: int = 20
    sentiment_default: float = 0.5


class FeatureEngineer:
    def __init__(self, config: Optional[FeatureEngineerConfig] = None) -> None:
        self.config = config or FeatureEngineerConfig()

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data.columns:
            raise ValueError("Expected a 'close' column for indicator calculations.")

        df = data.copy()
        df["ema"] = df["close"].ewm(span=self.config.ema_length, adjust=False).mean()
        if ta is not None:
            df["rsi"] = ta.rsi(df["close"], length=self.config.rsi_length)
        else:
            delta = df["close"].diff()
            gains = delta.clip(lower=0.0)
            losses = -delta.clip(upper=0.0)
            avg_gain = gains.rolling(self.config.rsi_length, min_periods=self.config.rsi_length).mean()
            avg_loss = losses.rolling(self.config.rsi_length, min_periods=self.config.rsi_length).mean()
            rs = avg_gain / avg_loss.replace(0.0, pd.NA)
            df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        return df

    def add_sentiment_score(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = float(self.config.sentiment_default)

        df["sentiment_score"] = df["sentiment_score"].astype(float).clip(0.0, 1.0)
        return df

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.add_indicators(data)
        df = self.add_sentiment_score(df)
        return df
