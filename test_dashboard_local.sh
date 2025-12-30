#!/bin/bash

# Local Testing Script for Streamlit Dashboard
# Run this before deploying to Azure to verify everything works

echo "=========================================="
echo "Testing Super Peter Daytrader Dashboard"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected"
    echo "Activating .venv..."
    source .venv/Scripts/activate  # Windows
    # source .venv/bin/activate    # Linux/Mac
fi

# Install package in development mode
echo "Installing trading_bot package..."
pip install -e . --quiet

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✓ Setup complete"
echo ""
echo "Starting Streamlit dashboard on http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

# Run Streamlit
streamlit run dashboard.py
