# Azure App Service Deployment Guide

## Overview

This guide explains how to deploy the Super Peter Daytrader Streamlit dashboard to Azure App Service (Linux).

## Prerequisites

- Azure account with active subscription
- Azure CLI installed ([Download](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
- Git installed
- Python 3.10+ installed locally

## Deployment Files Created

### 1. `requirements.txt`
Contains all Python dependencies:
- Core: numpy, pandas, python-dateutil
- Data: yfinance
- ML: scikit-learn, scipy, joblib
- Sentiment: textblob, vaderSentiment, nltk
- Web: streamlit
- Visualization: plotly, matplotlib

### 2. `startup.sh`
Bash script that launches the Streamlit app:
```bash
python -m streamlit run dashboard.py --server.port 8000 --server.address 0.0.0.0
```

**Critical**: Azure App Service (Linux) expects apps to listen on port **8000**, not Streamlit's default 8501.

### 3. `dashboard.py`
Streamlit application with:
- **Multiple file upload support** (`accept_multiple_files=True`)
- Data pipeline integration
- AI model training interface
- Prediction dashboard
- Interactive visualizations

### 4. `.streamlit/config.toml`
Streamlit configuration optimized for Azure:
- Server port: 8000
- Server address: 0.0.0.0
- Headless mode enabled
- CORS disabled for security

### 5. `.deployment`
Azure deployment configuration

## Step-by-Step Deployment

### Step 1: Prepare Your Repository

```bash
# Navigate to your project directory
cd C:\Users\johnn\PycharmProjects\Super_Peter_Daytrader

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Prepare for Azure deployment"
```

### Step 2: Login to Azure

```bash
# Login
az login

# Set subscription (if you have multiple)
az account set --subscription "Your-Subscription-Name"
```

### Step 3: Create Resource Group

```bash
# Create resource group
az group create --name SuperPeterDaytrader-RG --location eastus
```

### Step 4: Create App Service Plan

```bash
# Create Linux App Service Plan (B1 tier - suitable for development)
az appservice plan create \
    --name SuperPeterDaytrader-Plan \
    --resource-group SuperPeterDaytrader-RG \
    --sku B1 \
    --is-linux
```

**Pricing Tiers:**
- **F1** (Free): Limited, no custom domains
- **B1** (Basic): $13/month, good for development
- **P1V2** (Production): Better performance, $73/month

### Step 5: Create Web App

```bash
# Create web app with Python 3.10
az webapp create \
    --resource-group SuperPeterDaytrader-RG \
    --plan SuperPeterDaytrader-Plan \
    --name super-peter-daytrader \
    --runtime "PYTHON:3.10"
```

**Note**: The name `super-peter-daytrader` must be globally unique. If taken, try `super-peter-daytrader-yourname`.

### Step 6: Configure Startup Command

```bash
# Set startup command
az webapp config set \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --startup-file startup.sh
```

### Step 7: Configure App Settings

```bash
# Enable HTTP logging
az webapp log config \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --application-logging filesystem \
    --level verbose

# Set port configuration
az webapp config appsettings set \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --settings WEBSITES_PORT=8000
```

### Step 8: Deploy from Local Git

```bash
# Set up local git deployment
az webapp deployment source config-local-git \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader

# Get deployment credentials
az webapp deployment list-publishing-credentials \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --query "{username:publishingUserName, password:publishingPassword}"

# Add Azure remote (replace with your app name)
git remote add azure https://super-peter-daytrader.scm.azurewebsites.net/super-peter-daytrader.git

# Deploy
git push azure main
```

**Alternative: Deploy from GitHub**

```bash
# If your code is on GitHub
az webapp deployment source config \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --repo-url https://github.com/yourusername/super-peter-daytrader \
    --branch main \
    --manual-integration
```

### Step 9: Monitor Deployment

```bash
# Stream logs
az webapp log tail \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader
```

### Step 10: Access Your App

Your app will be available at:
```
https://super-peter-daytrader.azurewebsites.net
```

## Troubleshooting

### Issue 1: Application Not Starting

**Check logs:**
```bash
az webapp log tail --resource-group SuperPeterDaytrader-RG --name super-peter-daytrader
```

**Common causes:**
- Startup script not executable: `chmod +x startup.sh`
- Port mismatch: Ensure port 8000 is used
- Missing dependencies: Check requirements.txt

### Issue 2: Port 8000 Not Working

**Verify startup command:**
```bash
az webapp config show \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --query "appCommandLine"
```

**Should return:** `startup.sh`

### Issue 3: Import Errors

**Ensure package is installed:**
```bash
# Add to startup.sh
pip install -e .
```

### Issue 4: File Upload Not Working

**Check App Service file size limits:**
- Default: 30 MB per file
- To increase: Configure in Azure Portal → Configuration → General Settings

## Performance Optimization

### 1. Enable Application Insights

```bash
az monitor app-insights component create \
    --app super-peter-daytrader-insights \
    --location eastus \
    --resource-group SuperPeterDaytrader-RG \
    --application-type web
```

### 2. Enable Always On (Prevents cold starts)

```bash
az webapp config set \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --always-on true
```

**Note**: Always On requires Basic tier or higher (not available in Free tier)

### 3. Scale Up/Out

**Scale Up (Vertical):**
```bash
az appservice plan update \
    --resource-group SuperPeterDaytrader-RG \
    --name SuperPeterDaytrader-Plan \
    --sku P1V2
```

**Scale Out (Horizontal):**
```bash
az appservice plan update \
    --resource-group SuperPeterDaytrader-RG \
    --name SuperPeterDaytrader-Plan \
    --number-of-workers 3
```

## Security Best Practices

### 1. Enable HTTPS Only

```bash
az webapp update \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --https-only true
```

### 2. Configure Authentication (Optional)

```bash
az webapp auth update \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --enabled true \
    --action LoginWithAzureActiveDirectory
```

### 3. Restrict Access by IP (Optional)

```bash
az webapp config access-restriction add \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --rule-name "Office" \
    --priority 100 \
    --ip-address "YOUR.IP.ADDRESS.HERE/32"
```

## Custom Domain (Optional)

### 1. Add Custom Domain

```bash
az webapp config hostname add \
    --resource-group SuperPeterDaytrader-RG \
    --webapp-name super-peter-daytrader \
    --hostname www.yourdomain.com
```

### 2. Enable SSL

```bash
az webapp config ssl bind \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --certificate-thumbprint <thumbprint> \
    --ssl-type SNI
```

## Cost Management

### Estimated Monthly Costs

| Tier | Price/Month | Suitable For |
|------|-------------|--------------|
| F1 (Free) | $0 | Testing only |
| B1 (Basic) | ~$13 | Development |
| B2 (Basic) | ~$26 | Small production |
| P1V2 (Premium) | ~$73 | Production |
| P2V2 (Premium) | ~$146 | High traffic |

### Cost Optimization Tips

1. **Use Free tier for testing**
2. **Stop app when not in use:**
   ```bash
   az webapp stop --resource-group SuperPeterDaytrader-RG --name super-peter-daytrader
   ```
3. **Use auto-scaling based on load**
4. **Monitor usage with Azure Cost Management**

## Cleanup (Delete Resources)

To avoid charges, delete resources when done:

```bash
# Delete entire resource group (removes all resources)
az group delete --name SuperPeterDaytrader-RG --yes --no-wait
```

## Environment Variables

Set environment variables for configuration:

```bash
az webapp config appsettings set \
    --resource-group SuperPeterDaytrader-RG \
    --name super-peter-daytrader \
    --settings \
        MODEL_PATH="/home/site/wwwroot/models" \
        DATA_PATH="/home/site/wwwroot/data" \
        LOG_LEVEL="INFO"
```

## CI/CD with GitHub Actions

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Deploy to Azure
        uses: azure/webapps-deploy@v2
        with:
          app-name: 'super-peter-daytrader'
          publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
```

## Support

For issues:
1. Check Azure logs: `az webapp log tail ...`
2. Review deployment status in Azure Portal
3. Verify startup.sh is executable
4. Ensure port 8000 is configured correctly

## Additional Resources

- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy)
- [Azure CLI Reference](https://docs.microsoft.com/en-us/cli/azure/)
