import streamlit as st
import os
from azure.storage.blob import BlobServiceClient
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Super Peter - Cloud Upload", page_icon="☁️", layout="centered")

# Header
st.title("☁️ Super Peter Trade Uploader")
st.markdown("### Upload your trade logs here for processing.")

# 1. Setup Connection to "Database" (Blob Storage)
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
container_name = "trade-uploads"

# Check if connection string is available
if not connect_str:
    st.error("⚠️ Azure Storage Connection String not found.")
    st.info("Please go to Azure Portal -> App Service -> Settings -> Environment Variables and add 'AZURE_STORAGE_CONNECTION_STRING'.")
    st.stop()

# 2. File Uploader
uploaded_files = st.file_uploader(
    "Drag and drop Orders.csv here", 
    type=['csv'], 
    accept_multiple_files=True
)

# 3. Upload Logic
if st.button("🚀 Upload to Cloud Database") and uploaded_files:
    try:
        # Connect to the container
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
        # Create container if it doesn't exist (safety check)
        try:
            container_client = blob_service_client.create_container(container_name)
        except:
            container_client = blob_service_client.get_container_client(container_name)

        progress_bar = st.progress(0)
        
        for idx, file in enumerate(uploaded_files):
            # Create a unique name: "2025-12-30_Orders.csv"
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            blob_name = f"{timestamp}_{file.name}"
            
            # Upload
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(file, overwrite=True)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            st.success(f"✅ Successfully uploaded: {file.name}")

        st.balloons()
        st.info("Files are now safe in the cloud. You can run the trainer on your local machine.")

    except Exception as e:
        st.error(f"Upload failed: {e}")