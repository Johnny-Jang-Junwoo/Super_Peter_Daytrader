"""
Fetch Data from Azure Cloud Mailbox

This script downloads CSV files uploaded to Azure Blob Storage
so they can be processed locally with the heavy ML libraries.
"""

import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
CONNECT_STR = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = "trade-uploads"
LOCAL_INBOX = "data_pipeline/incoming"


def fetch_from_cloud():
    """Download new CSV files from Azure Blob Storage to local machine."""

    if not CONNECT_STR:
        print("❌ Error: AZURE_STORAGE_CONNECTION_STRING not found.")
        print("💡 Tip: Create a .env file with your Azure connection string")
        print("   (See .env.example for template)")
        return

    print(f"🔌 Connecting to Cloud Mailbox ({CONTAINER_NAME})...")

    try:
        # Connect to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        # Create local inbox directory if it doesn't exist
        if not os.path.exists(LOCAL_INBOX):
            os.makedirs(LOCAL_INBOX)
            print(f"📁 Created local inbox: {LOCAL_INBOX}")

        # List all blobs in the container
        blobs = container_client.list_blobs()
        new_files = 0
        skipped_files = 0

        for blob in blobs:
            local_path = os.path.join(LOCAL_INBOX, blob.name)

            # Only download if file doesn't exist locally
            if not os.path.exists(local_path):
                print(f"📥 Downloading: {blob.name}")
                with open(local_path, "wb") as f:
                    data = container_client.download_blob(blob.name).readall()
                    f.write(data)
                new_files += 1
            else:
                skipped_files += 1

        # Summary
        print("\n" + "="*60)
        if new_files > 0:
            print(f"✅ Downloaded {new_files} new file(s) to '{LOCAL_INBOX}'")
        else:
            print("💤 No new files found in the cloud.")

        if skipped_files > 0:
            print(f"ℹ️  Skipped {skipped_files} file(s) already on disk")

        print("="*60 + "\n")

    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your AZURE_STORAGE_CONNECTION_STRING is correct")
        print("   2. Verify the container 'trade-uploads' exists in Azure")
        print("   3. Check your network connection")


def list_local_files():
    """List files currently in the local inbox."""

    if not os.path.exists(LOCAL_INBOX):
        print(f"📁 Inbox '{LOCAL_INBOX}' is empty (not created yet)")
        return []

    files = [f for f in os.listdir(LOCAL_INBOX) if f.endswith('.csv')]

    if files:
        print(f"\n📋 Local inbox contains {len(files)} CSV file(s):")
        for f in files:
            file_path = os.path.join(LOCAL_INBOX, f)
            size = os.path.getsize(file_path)
            print(f"   • {f} ({size:,} bytes)")
    else:
        print(f"\n📁 Inbox '{LOCAL_INBOX}' is empty")

    return files


if __name__ == "__main__":
    print("=" * 60)
    print("🌩️  SUPER PETER CLOUD MAILBOX - DATA FETCHER")
    print("=" * 60)

    # Show current local files
    list_local_files()

    # Fetch new files from cloud
    fetch_from_cloud()

    # Show updated local files
    print("\n📊 After download:")
    list_local_files()
