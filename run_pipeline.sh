#!/bin/bash

# Super Peter Local Training Pipeline
# This script runs the complete workflow: fetch data → train model

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     🤖 SUPER PETER LOCAL TRAINING PIPELINE 🤖             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Fetch data from Azure
echo "📥 STEP 1: Fetching data from Azure Cloud Mailbox..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python fetch_data.py

# Check if fetch was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to fetch data from Azure"
    echo "💡 Check your .env file and AZURE_STORAGE_CONNECTION_STRING"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 2: Train model
echo "🤖 STEP 2: Training AI model on downloaded data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python train_local.py

# Check if training was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed"
    echo "💡 Check the error messages above for details"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  ✅ PIPELINE COMPLETE! ✅                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "✨ Your AI model has been trained and saved!"
echo "📁 Check the 'models/' folder for your trained model"
echo ""
