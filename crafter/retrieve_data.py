#!/usr/bin/env python3
"""
Simple script to retrieve data from Google Cloud Storage.
Run this script to get started with data retrieval.
"""

import sys
import os

# Add the data_analysis directory to the path
sys.path.append('data_analysis')

from data_analysis.latency import GoogleCloudDataRetriever, download_latency_data, analyze_latency_data
from config import BUCKET_DIR

def main():
    """Main function to demonstrate data retrieval."""
    
    # Your bucket name from the credentials
    BUCKET_NAME = "ishaanlatencytesting"
    
    print("🔍 Google Cloud Storage Data Retrieval")
    print("=" * 50)
    
    try:
        # Initialize the retriever
        print(f"Connecting to bucket: {BUCKET_NAME}")
        retriever = GoogleCloudDataRetriever(BUCKET_NAME)
        
        # List files in your bucket directory
        print(f"\n📁 Listing files in directory: {BUCKET_DIR}")
        files = retriever.list_files(prefix=BUCKET_DIR)
        
        if not files:
            print("No files found in the specified directory.")
            return
        
        print(f"Found {len(files)} files:")
        for i, file in enumerate(files[:10], 1):  # Show first 10 files
            print(f"  {i}. {file}")
        
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        
        # Ask user what they want to do
        print("\n" + "=" * 50)
        print("What would you like to do?")
        print("1. Download all latency data files")
        print("2. Analyze latency data")
        print("3. Create a DataFrame for analysis")
        print("4. List files in a different directory")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n📥 Downloading latency data...")
            local_dir = "downloaded_latency_data"
            downloaded_files = download_latency_data(BUCKET_NAME, local_dir)
            print(f"✅ Downloaded {len(downloaded_files)} files to '{local_dir}' directory")
            
        elif choice == "2":
            print("\n📊 Analyzing latency data...")
            analysis = analyze_latency_data(BUCKET_NAME)
            print("Analysis Results:")
            for key, value in analysis.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
                    
        elif choice == "3":
            print("\n📋 Creating DataFrame...")
            df = retriever.create_latency_dataframe()
            if not df.empty:
                print(f"✅ DataFrame created with shape: {df.shape}")
                print("\nFirst few rows:")
                print(df.head())
                
                # Save to CSV
                csv_file = "latency_data.csv"
                df.to_csv(csv_file, index=False)
                print(f"💾 DataFrame saved to '{csv_file}'")
            else:
                print("❌ No data found to create DataFrame")
                
        elif choice == "4":
            prefix = input("Enter directory prefix to list: ").strip()
            if prefix:
                print(f"\n📁 Listing files in directory: {prefix}")
                files = retriever.list_files(prefix=prefix)
                for file in files:
                    print(f"  {file}")
            else:
                print("❌ No prefix provided")
                
        elif choice == "5":
            print("👋 Goodbye!")
            return
            
        else:
            print("❌ Invalid choice. Please enter a number between 1-5.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your Google Cloud credentials are correct")
        print("2. Verify the bucket name is correct")
        print("3. Check that you have the required permissions")
        print("4. Ensure the google-cloud-storage package is installed")


if __name__ == "__main__":
    main() 