#!/bin/bash

# Azure App Service Startup Script
# We ONLY run Streamlit. We do NOT install the heavy trading_bot package.

echo "Starting Lightweight Mailbox App..."
echo "Port: 8000"
echo "Address: 0.0.0.0"

# REMOVED: pip install -e . (This was the cause of the crash)

python -m streamlit run dashboard.py --server.port 8000 --server.address 0.0.0.0