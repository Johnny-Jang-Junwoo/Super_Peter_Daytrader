from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .risk_manager import RiskManager


class StrategyBase(ABC):
    def __init__(
        self,
        data_loader: DataLoader,
        risk_manager: RiskManager,
        feature_engineer: Optional[FeatureEngineer] = None,
    ) -> None:
        self.data_loader = data_loader
        self.risk_manager = risk_manager
        self.feature_engineer = feature_engineer

    def can_trade(self) -> bool:
        return self.risk_manager.can_trade()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def size_position(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
