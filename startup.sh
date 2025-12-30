#!/bin/bash

# Azure App Service Startup Script for Streamlit
# This script launches the Streamlit dashboard on port 8000

echo "Starting Streamlit application..."
echo "Port: 8000"
echo "Address: 0.0.0.0"

# Install the trading_bot package in development mode
pip install -e .

# Launch Streamlit on port 8000 (Azure default)
# Note: Azure App Service (Linux) expects apps to listen on port 8000
python -m streamlit run dashboard.py --server.port 8000 --server.address 0.0.0.0
