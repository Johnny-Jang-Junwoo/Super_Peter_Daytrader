# Azure Deployment Troubleshooting Guide

## Problem: Site shows "deployed" but returns "cannot connect to site"

---

## Quick Diagnosis Commands

### 1. Check Application Logs (CRITICAL - Do This First!)
**Azure Portal Method:**
1. Go to Azure Portal → SuperPeterIntelligence App Service
2. Left menu → **Monitoring** → **Log stream**
3. Look for errors (Python crashes, import errors, etc.)

**OR use Azure CLI:**
```bash
# View real-time logs
az webapp log tail --name SuperPeterIntelligence --resource-group <your-resource-group>

# Download recent logs
az webapp log download --name SuperPeterIntelligence --resource-group <your-resource-group> --log-file logs.zip
```

---

## Most Common Issues

### Issue 1: Missing Environment Variable ⚠️ (MOST LIKELY)

**Symptom:** App crashes immediately on startup because `AZURE_STORAGE_CONNECTION_STRING` is not found.

**Check:**
```bash
az webapp config appsettings list --name SuperPeterIntelligence --resource-group <your-resource-group>
```

**Fix in Azure Portal:**
1. Go to SuperPeterIntelligence App Service
2. Left menu → **Settings** → **Environment variables**
3. Click **+ Add** under "App settings"
4. Add:
   - **Name:** `AZURE_STORAGE_CONNECTION_STRING`
   - **Value:** Your connection string from Storage Account → Access Keys
5. Click **Apply** → **Confirm**
6. **Restart the app**: Overview → **Restart**

**Fix with Azure CLI:**
```bash
az webapp config appsettings set \
  --name SuperPeterIntelligence \
  --resource-group <your-resource-group> \
  --settings AZURE_STORAGE_CONNECTION_STRING="<your-connection-string>"
```

---

### Issue 2: Startup Command Not Configured

**Symptom:** Azure isn't running your startup.sh script.

**Check current startup command:**
```bash
az webapp config show --name SuperPeterIntelligence --resource-group <your-resource-group> --query "appCommandLine"
```

**Fix in Azure Portal:**
1. SuperPeterIntelligence → **Settings** → **Configuration**
2. **General settings** tab
3. **Startup Command** field, enter:
   ```
   bash startup.sh
   ```
4. Click **Save**
5. **Restart the app**

**Fix with Azure CLI:**
```bash
az webapp config set \
  --name SuperPeterIntelligence \
  --resource-group <your-resource-group> \
  --startup-file "bash startup.sh"
```

---

### Issue 3: Python Version Mismatch

**Check Python version on Azure:**
```bash
az webapp config show --name SuperPeterIntelligence --resource-group <your-resource-group> --query "linuxFxVersion"
```

Should return: `PYTHON|3.10`

**Fix if wrong version:**
```bash
az webapp config set \
  --name SuperPeterIntelligence \
  --resource-group <your-resource-group> \
  --linux-fx-version "PYTHON|3.10"
```

---

### Issue 4: Port Configuration

**Check if WEBSITES_PORT is set:**
```bash
az webapp config appsettings list --name SuperPeterIntelligence --resource-group <your-resource-group> | grep WEBSITES_PORT
```

**If not found, add it:**
```bash
az webapp config appsettings set \
  --name SuperPeterIntelligence \
  --resource-group <your-resource-group> \
  --settings WEBSITES_PORT=8000
```

---

### Issue 5: Build Cache Problem

**Clear deployment cache and rebuild:**

**Azure Portal:**
1. SuperPeterIntelligence → **Development Tools** → **Advanced Tools**
2. Click **Go** (opens Kudu)
3. Top menu → **Tools** → **Delete Deployment Cache**

**OR via CLI:**
```bash
# Restart the app service
az webapp restart --name SuperPeterIntelligence --resource-group <your-resource-group>

# If that doesn't work, stop and start
az webapp stop --name SuperPeterIntelligence --resource-group <your-resource-group>
az webapp start --name SuperPeterIntelligence --resource-group <your-resource-group>
```

---

## Step-by-Step Diagnostic Workflow

### Step 1: Check Logs First
```bash
az webapp log tail --name SuperPeterIntelligence --resource-group <your-resource-group>
```
**Look for:**
- `ModuleNotFoundError` (missing dependencies)
- `KeyError: 'AZURE_STORAGE_CONNECTION_STRING'` (missing env var)
- Port binding errors
- Streamlit startup messages

---

### Step 2: Verify Environment Variables
**Required variable:** `AZURE_STORAGE_CONNECTION_STRING`

```bash
az webapp config appsettings list --name SuperPeterIntelligence --resource-group <your-resource-group>
```

If missing, add it (see Issue 1 above).

---

### Step 3: Check Startup Command
```bash
az webapp config show --name SuperPeterIntelligence --resource-group <your-resource-group> --query "appCommandLine"
```

Should return: `bash startup.sh` or `startup.sh`

If empty or wrong, set it (see Issue 2 above).

---

### Step 4: Verify Python Version and Runtime
```bash
az webapp config show --name SuperPeterIntelligence --resource-group <your-resource-group>
```

Check:
- `linuxFxVersion` = `PYTHON|3.10`
- `appCommandLine` = `bash startup.sh`

---

### Step 5: Test SSH into Container (Advanced)
```bash
az webapp ssh --name SuperPeterIntelligence --resource-group <your-resource-group>
```

Once inside:
```bash
# Check if files are deployed
ls -la

# Check if startup.sh exists
cat startup.sh

# Try running manually
bash startup.sh
```

---

## Expected Successful Logs

When working correctly, you should see:
```
Starting Lightweight Mailbox App...
Port: 8000
Address: 0.0.0.0

You can now view your Streamlit app in your browser.

Network URL: http://0.0.0.0:8000
External URL: http://<your-ip>:8000
```

---

## Quick Fix Checklist

- [ ] Environment variable `AZURE_STORAGE_CONNECTION_STRING` is set
- [ ] Startup command is `bash startup.sh`
- [ ] Python version is 3.10
- [ ] `WEBSITES_PORT=8000` is set (optional but recommended)
- [ ] Checked application logs for errors
- [ ] Restarted the app service after making changes

---

## Get Your Resource Group Name

If you don't know your resource group:
```bash
az webapp list --query "[?name=='SuperPeterIntelligence'].resourceGroup" -o tsv
```

---

## Emergency: Force Redeploy

If all else fails, trigger a fresh deployment:

**Method 1: Commit and Push**
```bash
# Make a trivial change to force redeploy
echo "# Force redeploy" >> README.md
git add .
git commit -m "Force redeploy"
git push origin main
```

**Method 2: Manual Redeploy via GitHub Actions**
1. Go to GitHub repository
2. **Actions** tab
3. Click on "Build and deploy Python app to Azure Web App"
4. Click **Run workflow** → **Run workflow**

---

## Still Not Working?

**Check GitHub Actions deployment status:**
1. Go to your GitHub repository
2. Click **Actions** tab
3. Look for failed workflows (red X)
4. Click on the latest run to see error details

**Common GitHub Actions errors:**
- Build failed (dependency issues)
- Authentication failed (Azure credentials expired)
- Deployment timeout

---

## Contact Information

If you're stuck, Azure Portal provides:
- **Resource Health** (left menu) - shows platform issues
- **Diagnose and solve problems** - automated diagnostics
- **Support + troubleshooting** - create support ticket

---

## Pro Tip: Enable Application Insights

For better debugging in the future:
1. SuperPeterIntelligence → **Settings** → **Application Insights**
2. Turn on Application Insights
3. This gives you detailed error tracking and performance monitoring
