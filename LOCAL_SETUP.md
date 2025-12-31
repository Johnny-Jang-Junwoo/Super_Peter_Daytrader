# Local AI Training Setup

## Overview

Your project is now split into two parts:

1. **☁️ Azure Cloud App** - Lightweight "mailbox" for file uploads (no ML libraries)
2. **💻 Local Machine** - Heavy AI training with full ML capabilities

## Architecture

```
Friend's Computer              Azure Cloud (Free Tier)          Your Local Machine
    📊 CSV Files    →    ☁️ Blob Storage (Mailbox)    →    🤖 AI Training
                         (lightweight uploader)            (heavy ML libraries)
```

## Setup Instructions

### 1. Install Local Dependencies

On your **local machine**, install the heavy ML libraries:

```bash
pip install -r requirements-local.txt
```

**Note:** Do NOT install these on Azure - they will crash the Free Tier!

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your Azure Storage connection string
# Get this from: Azure Portal → Storage Account → Access Keys
```

Your `.env` should look like:
```
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=yourname;AccountKey=yourkey;EndpointSuffix=core.windows.net
```

### 3. Fetch Data from Cloud

Download CSV files uploaded to Azure:

```bash
python fetch_data.py
```

**Output:**
```
🔌 Connecting to Cloud Mailbox (trade-uploads)...
📥 Downloading: 2024-12-30_Orders.csv
✅ Downloaded 1 new file(s) to 'data_pipeline/incoming'
```

Files are saved to: `data_pipeline/incoming/`

### 4. Train AI Model

Run the training pipeline:

```bash
python train_local.py
```

**What it does:**
1. ✅ Loads CSV files from inbox
2. ✅ Fetches market data from yfinance
3. ✅ Merges trades with market candles
4. ✅ Adds technical indicators (RSI, EMA, etc.)
5. ✅ Trains Random Forest model
6. ✅ Saves model to `models/`
7. ✅ Archives processed files

**Output:**
```
🤖 SUPER PETER LOCAL AI TRAINER
========================================
📁 Found 1 CSV file(s) to process
✅ Total trades combined: 28
📊 Retrieved 345 market candles
✅ Training set ready: 331 samples
✅ Model saved to: models/behavioral_cloner_MNQ_20241230_143022.pkl
🎉 TRAINING COMPLETE!
```

## Workflow

### Daily Routine

1. **Friend uploads CSV** → Azure Cloud App (☁️ Mailbox)
2. **You fetch data** → Run `python fetch_data.py`
3. **You train model** → Run `python train_local.py`
4. **Model ready** → Use for predictions locally

### File Locations

```
data_pipeline/
├── incoming/        ← Downloaded CSV files (fresh from cloud)
└── processed/       ← Archived files after training

models/
└── behavioral_cloner_*.pkl   ← Trained AI models
```

## Scripts Reference

### `fetch_data.py`
**Purpose:** Download CSV files from Azure Blob Storage

**Usage:**
```bash
python fetch_data.py
```

**Features:**
- ✅ Downloads new files only (skips existing)
- ✅ Shows download progress
- ✅ Creates local inbox directory automatically
- ✅ Lists local files before/after

### `train_local.py`
**Purpose:** Train AI model on downloaded data

**Usage:**
```bash
python train_local.py
```

**Features:**
- ✅ Processes all CSV files in inbox
- ✅ Fetches market data (1-minute OHLCV)
- ✅ Adds technical indicators
- ✅ Trains Random Forest classifier
- ✅ Saves model with timestamp
- ✅ Archives processed files

### `.env.example`
**Purpose:** Template for environment variables

**Setup:**
```bash
cp .env.example .env
# Edit .env with your actual Azure connection string
```

## Troubleshooting

### Issue: "AZURE_STORAGE_CONNECTION_STRING not found"

**Solution:**
1. Create `.env` file: `cp .env.example .env`
2. Get connection string from Azure Portal:
   - Storage Account → Access Keys → Connection String
3. Paste into `.env` file

### Issue: "No CSV files found in data_pipeline/incoming"

**Solution:**
1. Run `python fetch_data.py` first to download files
2. Check Azure Blob Storage has files uploaded
3. Verify container name is "trade-uploads"

### Issue: "No market data available from yfinance"

**Cause:** 1-minute data only available for last 7-30 days

**Solutions:**
- Use recent trade dates (within last week)
- Script will create synthetic data for demonstration
- For historical data, modify to use daily interval

### Issue: "Training failed - insufficient data"

**Solution:**
- Need at least 10 samples with some buy/sell signals
- Upload more CSV files
- Check trades are being matched to market candles

## Advanced Usage

### Batch Processing

Process multiple days of data:

```bash
# Fetch all new files
python fetch_data.py

# Train on everything
python train_local.py
```

### Custom Configuration

Edit `train_local.py` to customize:
- Model parameters (n_estimators, max_depth)
- Feature engineering (add custom indicators)
- Data processing (different intervals)

### Using Trained Models

```python
from trading_bot import BehavioralCloner

# Load trained model
cloner = BehavioralCloner()
cloner.load_brain("models/behavioral_cloner_MNQ_20241230.pkl")

# Make predictions
predictions = cloner.predict(X)
```

## Security Notes

### ⚠️ Important: .env File

- **NEVER commit `.env` to git** (already in `.gitignore`)
- Contains sensitive Azure credentials
- Each developer needs their own `.env`

### Azure Connection String

- Keep it secret (like a password)
- Rotate periodically in Azure Portal
- Don't share in chat/email

## Performance

### Local vs Cloud

| Feature | Azure Cloud | Local Machine |
|---------|-------------|---------------|
| Purpose | File upload mailbox | AI training |
| RAM Usage | ~100 MB | 1-4 GB |
| CPU Usage | Minimal | Heavy |
| Libraries | 2 (streamlit, azure-storage-blob) | 15+ (includes scikit-learn) |
| Cost | Free Tier | Your electricity 😊 |

## Next Steps

1. ✅ Install local dependencies: `pip install -r requirements-local.txt`
2. ✅ Create `.env` file with Azure credentials
3. ✅ Run `python fetch_data.py` to download files
4. ✅ Run `python train_local.py` to train models
5. ✅ Check `models/` folder for trained models
6. 🎯 Use models for predictions!

---

**Questions?** Check the main `README.md` or documentation in `docs/` folder.
