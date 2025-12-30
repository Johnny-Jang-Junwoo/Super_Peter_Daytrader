"""
AI Trainer Module for Behavioral Cloning Trading Bot

This module implements behavioral cloning to learn trading strategies
from historical trade logs using supervised learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass
class BehavioralClonerConfig:
    """Configuration for the Behavioral Cloner."""

    n_estimators: int = 100
    max_depth: Optional[int] = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    test_size: float = 0.2
    class_weight: str = "balanced"  # Handle imbalanced data


class BehavioralCloner:
    """
    Behavioral Cloning AI that learns trading decisions from expert trade logs.

    This class uses Random Forest Classification to learn when to buy, sell,
    or hold based on market features and historical trading decisions.
    """

    def __init__(self, config: Optional[BehavioralClonerConfig] = None) -> None:
        """
        Initialize the Behavioral Cloner.

        Args:
            config: Configuration for the model. If None, uses defaults.
        """
        self.config = config or BehavioralClonerConfig()
        self.model: Optional[RandomForestClassifier] = None
        self.feature_columns: Optional[list[str]] = None
        self.is_trained = False

    def prepare_labels(
        self,
        market_data: pd.DataFrame,
        trade_logs: pd.DataFrame | str | Path,
    ) -> pd.DataFrame:
        """
        Merge market data with trade logs to create labeled training data.

        Args:
            market_data: DataFrame with market data (must have 'date' column)
            trade_logs: Either a DataFrame or path to CSV file with trade logs.
                       Expected columns: 'date', 'action' (where action is 'buy', 'sell', or similar)

        Returns:
            DataFrame with market data and 'target' column (1=Buy, -1=Sell, 0=Hold)
        """
        # Load trade logs if path is provided
        if isinstance(trade_logs, (str, Path)):
            trade_logs = pd.read_csv(trade_logs)

        # Ensure we have a copy of market data
        df = market_data.copy()

        # Ensure date columns are datetime and normalized
        if "date" not in df.columns:
            raise ValueError("market_data must have a 'date' column")

        # Check if market data has timezone-aware dates
        has_tz = pd.api.types.is_datetime64tz_dtype(df["date"])

        if has_tz:
            # Keep timezone info but normalize
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        else:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        # Process trade logs
        if "date" not in trade_logs.columns:
            raise ValueError("trade_logs must have a 'date' column")

        if "action" not in trade_logs.columns:
            raise ValueError("trade_logs must have an 'action' column")

        # Normalize trade log dates and match timezone
        trade_logs = trade_logs.copy()
        trade_logs["date"] = pd.to_datetime(trade_logs["date"])

        # Match timezone of market data
        if has_tz:
            # Convert trade log dates to UTC to match market data
            if trade_logs["date"].dt.tz is None:
                trade_logs["date"] = trade_logs["date"].dt.tz_localize("UTC")
            else:
                trade_logs["date"] = trade_logs["date"].dt.tz_convert("UTC")

        trade_logs["date"] = trade_logs["date"].dt.normalize()

        # Map actions to numeric labels
        # Accepts various formats: 'buy', 'Buy', 'BUY', 'sell', 'Sell', 'SELL', etc.
        def map_action_to_label(action: str) -> int:
            action_lower = str(action).lower().strip()
            if action_lower in ["buy", "long", "enter_long", "1"]:
                return 1
            elif action_lower in ["sell", "short", "enter_short", "exit", "-1"]:
                return -1
            else:
                return 0

        trade_logs["target"] = trade_logs["action"].apply(map_action_to_label)

        # Keep only date and target from trade logs
        trade_labels = trade_logs[["date", "target"]].copy()

        # Remove any duplicate dates (keep first occurrence)
        trade_labels = trade_labels.drop_duplicates(subset=["date"], keep="first")

        # Merge with market data (left join to keep all market data)
        df = df.merge(trade_labels, on="date", how="left")

        # Fill missing targets with 0 (Hold)
        df["target"] = df["target"].fillna(0).astype(int)

        # Report class distribution
        class_counts = df["target"].value_counts().sort_index()
        print("\n=== Label Distribution ===")
        print(f"Hold (0):  {class_counts.get(0, 0):5d} samples")
        print(f"Buy (1):   {class_counts.get(1, 0):5d} samples")
        print(f"Sell (-1): {class_counts.get(-1, 0):5d} samples")

        total_trades = class_counts.get(1, 0) + class_counts.get(-1, 0)
        total_samples = len(df)
        print(f"\nTotal samples: {total_samples}")
        print(f"Total trades: {total_trades} ({100*total_trades/total_samples:.2f}%)")
        print("=========================\n")

        return df

    def train_model(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Train the Random Forest classifier on the prepared data.

        Args:
            X: Feature matrix (market indicators)
            y: Target labels (1=Buy, -1=Sell, 0=Hold)
            verbose: Whether to print training progress and metrics

        Returns:
            Dictionary with training metrics and evaluation results
        """
        # Store feature columns for later use
        if isinstance(X, pd.DataFrame):
            self.feature_columns = list(X.columns)
            X_array = X.values
        else:
            X_array = X

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_array,
            y_array,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_array,  # Maintain class distribution in splits
        )

        if verbose:
            print(f"Training set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}")
            print("\nTraining Random Forest Classifier...")

        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            class_weight=self.config.class_weight,  # Handle imbalanced classes
            n_jobs=-1,  # Use all CPU cores
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate accuracies
        train_accuracy = np.mean(y_train_pred == y_train)
        test_accuracy = np.mean(y_test_pred == y_test)

        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING RESULTS")
            print(f"{'='*60}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy:  {test_accuracy:.4f}")

            # Classification report
            print(f"\n{'='*60}")
            print("CLASSIFICATION REPORT (Test Set)")
            print(f"{'='*60}")
            target_names = ["Sell (-1)", "Hold (0)", "Buy (1)"]
            print(classification_report(y_test, y_test_pred, target_names=target_names))

            # Confusion Matrix
            print(f"{'='*60}")
            print("CONFUSION MATRIX (Test Set)")
            print(f"{'='*60}")
            cm = confusion_matrix(y_test, y_test_pred)
            print("           Predicted")
            print("           Sell  Hold  Buy")
            print(f"Actual Sell  {cm[0][0]:4d}  {cm[0][1]:4d}  {cm[0][2]:4d}")
            print(f"       Hold  {cm[1][0]:4d}  {cm[1][1]:4d}  {cm[1][2]:4d}")
            print(f"       Buy   {cm[2][0]:4d}  {cm[2][1]:4d}  {cm[2][2]:4d}")
            print(f"{'='*60}\n")

            # Feature importance
            if self.feature_columns:
                print(f"{'='*60}")
                print("TOP 10 FEATURE IMPORTANCES")
                print(f"{'='*60}")
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]
                for i, idx in enumerate(indices, 1):
                    feature_name = self.feature_columns[idx]
                    print(f"{i:2d}. {feature_name:20s}: {importances[idx]:.4f}")
                print(f"{'='*60}\n")

        # Return metrics dictionary
        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "classification_report": classification_report(
                y_test, y_test_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred),
        }

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions (1=Buy, -1=Sell, 0=Hold)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        return self.model.predict(X_array)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities for each class.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        return self.model.predict_proba(X_array)

    def save_brain(self, filename: str | Path) -> None:
        """
        Save the trained model to disk using joblib.

        Args:
            filename: Path where the model should be saved (e.g., 'model.pkl')
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Cannot save an untrained model")

        # Ensure the directory exists
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "config": self.config,
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved successfully to: {filepath}")

    def load_brain(self, filename: str | Path) -> None:
        """
        Load a trained model from disk.

        Args:
            filename: Path to the saved model file
        """
        filepath = Path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load model and metadata
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.config = model_data["config"]
        self.is_trained = True

        print(f"Model loaded successfully from: {filepath}")
        if self.feature_columns:
            print(f"Expected features: {self.feature_columns}")
