from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional


@dataclass
class RiskLimits:
    daily_max_loss: float
    total_drawdown: float


class RiskManager:
    def __init__(self, starting_equity: float, limits: RiskLimits) -> None:
        self.starting_equity = float(starting_equity)
        self.limits = limits
        self.current_equity = float(starting_equity)
        self.peak_equity = float(starting_equity)
        self.daily_start_equity = float(starting_equity)
        self.last_reset_date: Optional[date] = None

    def reset_daily(self, timestamp: Optional[datetime] = None) -> None:
        if timestamp is not None:
            self.last_reset_date = timestamp.date()
        self.daily_start_equity = float(self.current_equity)

    def update_equity(self, equity: float, timestamp: Optional[datetime] = None) -> None:
        self.current_equity = float(equity)
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        if timestamp is not None:
            if self.last_reset_date is None or timestamp.date() != self.last_reset_date:
                self.reset_daily(timestamp)

    def can_trade(self) -> bool:
        daily_loss = self.daily_start_equity - self.current_equity
        total_drawdown = self.peak_equity - self.current_equity
        if daily_loss >= self.limits.daily_max_loss:
            return False
        if total_drawdown >= self.limits.total_drawdown:
            return False
        return True
