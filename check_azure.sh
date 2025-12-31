#!/bin/bash

# Azure Deployment Diagnostic Script
# Run this to quickly check your Azure configuration

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        🔍 AZURE DEPLOYMENT DIAGNOSTICS 🔍                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

APP_NAME="SuperPeterIntelligence"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed"
    echo "💡 Install from: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

echo "✅ Azure CLI is installed"
echo ""

# Check if logged in
echo "🔐 Checking Azure login status..."
if ! az account show &> /dev/null; then
    echo "❌ Not logged in to Azure"
    echo "💡 Run: az login"
    exit 1
fi
echo "✅ Logged in to Azure"
echo ""

# Get resource group
echo "📁 Finding resource group..."
RESOURCE_GROUP=$(az webapp list --query "[?name=='$APP_NAME'].resourceGroup" -o tsv)

if [ -z "$RESOURCE_GROUP" ]; then
    echo "❌ Could not find app '$APP_NAME'"
    echo "💡 Check app name in Azure Portal"
    exit 1
fi

echo "✅ Found resource group: $RESOURCE_GROUP"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  CHECKING ENVIRONMENT VARIABLES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ENV_VARS=$(az webapp config appsettings list --name $APP_NAME --resource-group $RESOURCE_GROUP --query "[].name" -o tsv)

if echo "$ENV_VARS" | grep -q "AZURE_STORAGE_CONNECTION_STRING"; then
    echo "✅ AZURE_STORAGE_CONNECTION_STRING is set"
else
    echo "❌ AZURE_STORAGE_CONNECTION_STRING is NOT set"
    echo "💡 FIX: Run this command:"
    echo "   az webapp config appsettings set \\"
    echo "     --name $APP_NAME \\"
    echo "     --resource-group $RESOURCE_GROUP \\"
    echo "     --settings AZURE_STORAGE_CONNECTION_STRING=\"<your-connection-string>\""
    echo ""
fi

if echo "$ENV_VARS" | grep -q "WEBSITES_PORT"; then
    echo "✅ WEBSITES_PORT is set"
else
    echo "⚠️  WEBSITES_PORT is not set (optional but recommended)"
    echo "💡 FIX: Run this command:"
    echo "   az webapp config appsettings set \\"
    echo "     --name $APP_NAME \\"
    echo "     --resource-group $RESOURCE_GROUP \\"
    echo "     --settings WEBSITES_PORT=8000"
    echo ""
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  CHECKING STARTUP COMMAND"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STARTUP_CMD=$(az webapp config show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "appCommandLine" -o tsv)

if [ -z "$STARTUP_CMD" ] || [ "$STARTUP_CMD" == "None" ]; then
    echo "❌ Startup command is NOT set"
    echo "💡 FIX: Run this command:"
    echo "   az webapp config set \\"
    echo "     --name $APP_NAME \\"
    echo "     --resource-group $RESOURCE_GROUP \\"
    echo "     --startup-file \"bash startup.sh\""
    echo ""
else
    echo "✅ Startup command: $STARTUP_CMD"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  CHECKING PYTHON VERSION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PYTHON_VERSION=$(az webapp config show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "linuxFxVersion" -o tsv)

if [ "$PYTHON_VERSION" == "PYTHON|3.10" ]; then
    echo "✅ Python version: $PYTHON_VERSION"
else
    echo "⚠️  Python version: $PYTHON_VERSION"
    echo "💡 Expected: PYTHON|3.10"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  CHECKING APPLICATION STATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

APP_STATE=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "state" -o tsv)

if [ "$APP_STATE" == "Running" ]; then
    echo "✅ App state: $APP_STATE"
else
    echo "❌ App state: $APP_STATE"
    echo "💡 FIX: Restart the app:"
    echo "   az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  CHECKING APPLICATION URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

APP_URL=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "defaultHostName" -o tsv)
echo "🌐 App URL: https://$APP_URL"
echo ""
echo "Testing connection..."

if curl -s --max-time 10 "https://$APP_URL" > /dev/null; then
    echo "✅ Site is responding!"
else
    echo "❌ Site is NOT responding"
    echo "💡 Check application logs (see below)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6️⃣  NEXT STEPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 To view real-time application logs:"
echo "   az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "🔄 To restart the app after making fixes:"
echo "   az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "📥 To download full logs:"
echo "   az webapp log download --name $APP_NAME --resource-group $RESOURCE_GROUP --log-file logs.zip"
echo ""
echo "🌐 Open in browser:"
echo "   https://$APP_URL"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
