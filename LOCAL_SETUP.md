# Local AI Training Setup

## Overview

This project runs locally on Windows. Trade logs live in:

G:\My Drive\SuperPeterTrader

The training pipeline reads those CSV files and trains a behavioral cloning model.

## Setup

1. Install dependencies:
   pip install -r requirements-local.txt

2. Verify data access:
   python verify_local_data.py

3. Train the model:
   python train_local.py

## Data Requirements

- CSV files are expected. If you have Excel files, export them to CSV first.
- Expected columns include: Fill Time, Product, B/S, Status, and Exec Price.

## Output

- Trained models are saved in: models/

## Troubleshooting

- No CSV files found:
  - Confirm files exist in G:\My Drive\SuperPeterTrader
  - Or update GOOGLE_DRIVE_DIR in train_local.py

- Market data errors:
  - Verify internet access for yfinance
  - Confirm the symbol mapping in src/trading_bot/data_loader.py
