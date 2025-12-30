# AI Trainer (Behavioral Cloning) Guide

## Overview

The **BehavioralCloner** class implements supervised learning to train a trading bot that mimics an expert trader's decisions. It uses Random Forest Classification to learn when to Buy, Sell, or Hold based on market features and historical trade logs.

## Key Features

- **Behavioral Cloning**: Learn from expert trade logs
- **Imbalanced Data Handling**: Uses `class_weight='balanced'` to handle rare trading signals
- **Feature Importance**: Identifies which indicators matter most
- **Model Persistence**: Save and load trained models with joblib
- **Comprehensive Metrics**: Classification reports, confusion matrices, and feature importance

## Installation

The AI Trainer is included in the trading_bot package. Ensure scikit-learn is installed:

```bash
pip install scikit-learn
```

## Quick Start

```python
from trading_bot import (
    BehavioralCloner,
    BehavioralClonerConfig,
    DataLoader,
    FeatureEngineer,
)

# 1. Load and prepare market data
data_loader = DataLoader()
market_data = data_loader.load("AAPL")

# 2. Add technical indicators
feature_engineer = FeatureEngineer()
market_data = feature_engineer.transform(market_data)

# 3. Initialize the cloner
cloner = BehavioralCloner()

# 4. Prepare labels from trade logs
labeled_data = cloner.prepare_labels(market_data, "trade_logs.csv")

# 5. Train the model
X = labeled_data[["close", "volume", "rsi", "ema"]]
y = labeled_data["target"]
metrics = cloner.train_model(X, y)

# 6. Save the model
cloner.save_brain("models/my_model.pkl")

# 7. Make predictions
predictions = cloner.predict(X)
```

## Trade Log Format

Your trade log CSV should have the following structure:

```csv
date,action
2024-01-15,buy
2024-01-20,sell
2024-02-03,buy
2024-02-10,sell
```

### Supported Actions

The `action` column accepts various formats:
- **Buy signals**: `buy`, `Buy`, `BUY`, `long`, `enter_long`, `1`
- **Sell signals**: `sell`, `Sell`, `SELL`, `short`, `enter_short`, `exit`, `-1`
- **Hold signals**: Anything else (or omit the date entirely)

## Class Reference

### BehavioralClonerConfig

Configuration dataclass for the behavioral cloner.

```python
@dataclass
class BehavioralClonerConfig:
    n_estimators: int = 100          # Number of trees in the forest
    max_depth: Optional[int] = 10    # Maximum tree depth
    min_samples_split: int = 5       # Min samples to split a node
    min_samples_leaf: int = 2        # Min samples in a leaf node
    random_state: int = 42           # Random seed for reproducibility
    test_size: float = 0.2           # Proportion of data for testing
    class_weight: str = "balanced"   # Handle imbalanced classes
```

### BehavioralCloner

Main class for behavioral cloning.

#### Methods

##### `__init__(config: Optional[BehavioralClonerConfig] = None)`

Initialize the behavioral cloner.

**Parameters:**
- `config`: Configuration object. If None, uses defaults.

---

##### `prepare_labels(market_data: pd.DataFrame, trade_logs: pd.DataFrame | str | Path) -> pd.DataFrame`

Merge market data with trade logs to create labeled training data.

**Parameters:**
- `market_data`: DataFrame with market data (must have 'date' column)
- `trade_logs`: Either a DataFrame or path to CSV with trade logs

**Returns:**
- DataFrame with market data and 'target' column (1=Buy, -1=Sell, 0=Hold)

**Example:**
```python
labeled_data = cloner.prepare_labels(market_data, "trades.csv")
```

**Output:**
```
=== Label Distribution ===
Hold (0):    223 samples
Buy (1):       7 samples
Sell (-1):     7 samples

Total samples: 237
Total trades: 14 (5.91%)
=========================
```

---

##### `train_model(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, verbose: bool = True) -> dict`

Train the Random Forest classifier.

**Parameters:**
- `X`: Feature matrix (market indicators)
- `y`: Target labels (1=Buy, -1=Sell, 0=Hold)
- `verbose`: Whether to print training progress and metrics

**Returns:**
- Dictionary with training metrics

**Example:**
```python
X = labeled_data[["close", "rsi", "ema", "volume"]]
y = labeled_data["target"]
metrics = cloner.train_model(X, y)

print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
```

**Output Includes:**
- Training and test set sizes
- Training and test accuracy
- Classification report (Precision, Recall, F1-score)
- Confusion matrix
- Top 10 feature importances

---

##### `predict(X: pd.DataFrame | np.ndarray) -> np.ndarray`

Make predictions on new data.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Array of predictions (1=Buy, -1=Sell, 0=Hold)

**Example:**
```python
predictions = cloner.predict(new_data[features])
```

---

##### `predict_proba(X: pd.DataFrame | np.ndarray) -> np.ndarray`

Get prediction probabilities for each class.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Array of shape (n_samples, n_classes) with probabilities

**Example:**
```python
probabilities = cloner.predict_proba(new_data[features])
# probabilities[:, 0] = probability of Sell
# probabilities[:, 1] = probability of Hold
# probabilities[:, 2] = probability of Buy
```

---

##### `save_brain(filename: str | Path) -> None`

Save the trained model to disk.

**Parameters:**
- `filename`: Path where the model should be saved (e.g., 'model.pkl')

**Example:**
```python
cloner.save_brain("models/my_strategy.pkl")
```

---

##### `load_brain(filename: str | Path) -> None`

Load a trained model from disk.

**Parameters:**
- `filename`: Path to the saved model file

**Example:**
```python
new_cloner = BehavioralCloner()
new_cloner.load_brain("models/my_strategy.pkl")
predictions = new_cloner.predict(X)
```

## Handling Imbalanced Data

Trading decisions are naturally imbalanced - most of the time you're holding, not trading. The BehavioralCloner addresses this with:

1. **Class Weighting**: `class_weight='balanced'` automatically adjusts weights inversely proportional to class frequencies
2. **Stratified Splitting**: Maintains class distribution in train/test splits
3. **Comprehensive Metrics**: Precision/Recall/F1 for each class

### Best Practices

1. **Collect More Trade Signals**: The more buy/sell examples, the better the model learns
2. **Monitor Recall**: For trading, recall (catching actual opportunities) may be more important than precision
3. **Use Cross-Validation**: For small datasets, consider k-fold cross-validation
4. **Feature Engineering**: Add relevant technical indicators and derived features

## Feature Selection

The model automatically reports feature importances. Use this to:

1. Identify which indicators are most predictive
2. Remove low-importance features to reduce overfitting
3. Add related features that might improve predictions

Example output:
```
TOP 10 FEATURE IMPORTANCES
1. rsi                 : 0.2326
2. volume              : 0.1729
3. volume_change       : 0.1431
4. close               : 0.1330
5. high_low_spread     : 0.1247
```

## Common Issues and Solutions

### Issue: Low Test Accuracy

**Possible Causes:**
- Not enough training data
- Features don't contain predictive information
- Overfitting to training data

**Solutions:**
- Collect more trade logs
- Add more relevant technical indicators
- Reduce `max_depth` or increase `min_samples_leaf`
- Use cross-validation

### Issue: Model Predicts Only "Hold"

**Possible Causes:**
- Extremely imbalanced dataset
- Class weighting not effective enough

**Solutions:**
- Collect more buy/sell examples
- Try SMOTE (Synthetic Minority Over-sampling)
- Adjust `class_weight` manually
- Use different algorithms (e.g., XGBoost with scale_pos_weight)

### Issue: High Training Accuracy, Low Test Accuracy

**Cause:** Overfitting

**Solutions:**
- Reduce model complexity (`max_depth`, `n_estimators`)
- Increase `min_samples_split` and `min_samples_leaf`
- Add regularization
- Collect more training data

## Integration with Live Trading

To use the trained model in live trading:

```python
from trading_bot import BehavioralCloner, DataLoader, FeatureEngineer

# Load the trained model
cloner = BehavioralCloner()
cloner.load_brain("models/my_strategy.pkl")

# Fetch latest market data
data_loader = DataLoader()
current_data = data_loader.load("AAPL")

# Add features
feature_engineer = FeatureEngineer()
current_data = feature_engineer.transform(current_data)

# Get prediction for latest day
latest_features = current_data.tail(1)[feature_columns]
prediction = cloner.predict(latest_features)[0]
probabilities = cloner.predict_proba(latest_features)[0]

# Map to action
action_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
print(f"Signal: {action_map[prediction]}")
print(f"Confidence: {max(probabilities)*100:.1f}%")
```

## Advanced: Custom Strategies

Create a strategy class that uses the behavioral cloner:

```python
from trading_bot import StrategyBase
import pandas as pd

class AIStrategy(StrategyBase):
    def __init__(self, data_loader, risk_manager, cloner, features):
        super().__init__(data_loader, risk_manager, None)
        self.cloner = cloner
        self.features = features

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        X = df[self.features]
        df["signal"] = self.cloner.predict(X)
        return df

    def size_position(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Use prediction probabilities for position sizing
        X = df[self.features]
        probabilities = self.cloner.predict_proba(X)

        # Position size based on confidence
        max_probs = probabilities.max(axis=1)
        df["position_size"] = 0.0
        df.loc[df["signal"] != 0, "position_size"] = max_probs[df["signal"] != 0] * 0.1

        return df
```

## Performance Tips

1. **Feature Scaling**: Random Forests don't require feature scaling, but some features might benefit
2. **Feature Engineering**: Create domain-specific features (e.g., Disparity Index, Bollinger Bands)
3. **Ensemble Methods**: Combine multiple models for better predictions
4. **Time-Series Validation**: Use walk-forward validation for realistic performance estimates

## Next Steps

1. Run the example: `python examples/behavioral_cloning_example.py`
2. Prepare your real trade logs in the correct format
3. Experiment with different features and hyperparameters
4. Backtest the model on historical data
5. Implement paper trading before going live

## References

- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Handling Imbalanced Data](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
