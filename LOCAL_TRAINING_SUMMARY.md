# Local Training Summary

## Files

- verify_local_data.py: scans the local data folder and prints a summary.
- train_local.py: trains the model using the Google Drive data folder.
- requirements-local.txt: local dependencies for training and analysis.
- LOCAL_SETUP.md: setup and troubleshooting notes.

## Quick Start

1. pip install -r requirements-local.txt
2. python verify_local_data.py
3. python train_local.py

## Data Location

train_local.py reads CSV files from:
G:\My Drive\SuperPeterTrader

Update GOOGLE_DRIVE_DIR in train_local.py if your data lives elsewhere.

## Output

Trained models are saved under:
models/
