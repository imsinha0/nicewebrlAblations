import asyncio

from dotenv import load_dotenv
from google.auth.exceptions import TransportError
from google.cloud import storage
from google.api_core import exceptions as gcs_exceptions  # Fixed import

import json
import os
from datetime import datetime

from nicewebrl.logging import get_logger

from config import BUCKET_DIR

load_dotenv()
logger = get_logger(__name__)

GOOGLE_CREDENTIALS = "./google-cloud-key.json"


def initialize_storage_client(bucket_name: str):
  storage_client = storage.Client.from_service_account_json(GOOGLE_CREDENTIALS)

  bucket = storage_client.bucket(bucket_name)
  return bucket


def list_files(bucket):
  blobs = bucket.list_blobs()
  print("Files in bucket:")
  for blob in blobs:
    print(blob.name)


def download_files(bucket, destination_folder):
  blobs = bucket.list_blobs()
  if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

  for blob in blobs:
    file_path = os.path.join(destination_folder, blob.name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    blob.download_to_filename(file_path)
    print(f"Downloaded {blob.name} to {file_path}")


async def save_data_to_gcs(data, blob_filename, bucket_name: str):
  try:
    bucket = initialize_storage_client(bucket_name)
    blob = bucket.blob(blob_filename)

    # Run the blocking upload in a thread pool
    await asyncio.to_thread(
      blob.upload_from_string, data=json.dumps(data), content_type="application/json"
    )

    logger.info(f"Saved {blob_filename} in bucket {bucket.name}")
    return True  # Successfully saved
  except (TransportError, gcs_exceptions.GoogleAPIError) as e:  # Fixed exception
    logger.info(f"Error saving to GCS: {e}")
  except Exception as e:
    logger.info(f"Unexpected error: {e}")
    logger.info("Skipping GCS upload")

  return False  # Failed to save


async def save_file_to_gcs(local_filename, blob_filename, bucket_name: str):
  try:
    bucket = initialize_storage_client(bucket_name)
    blob = bucket.blob(blob_filename)

    # Run the blocking upload in a thread pool
    await asyncio.to_thread(blob.upload_from_filename, local_filename)

    logger.info(f"Saved {blob_filename} in bucket {bucket.name}")
    return True  # Successfully saved
  except Exception as e:
    logger.info(f"Unexpected error: {e}")
    logger.info("Skipping GCS upload")

  return False  # Failed to save


async def save_to_gcs_with_retries(
  files_to_save, max_retries=5, retry_delay=5, bucket_name: str = ""
):
  """Save multiple files to Google Cloud Storage with retry logic.

  Args:
      files_to_save: List of filenames
      max_retries: Number of retry attempts
      retry_delay: Seconds to wait between retries

  Returns:
      bool: True if all files were saved successfully, False otherwise
  """
  assert bucket_name != "", "Bucket name is required"
  for attempt in range(max_retries):
    try:
      # Try to save all files
      for local_file in files_to_save:
        saved = await save_file_to_gcs(
          local_filename=local_file, blob_filename=local_file, bucket_name=bucket_name
        )
        if not saved:
          raise Exception(f"Failed to save {local_file}")

      logger.info(f"Successfully saved data to GCS on attempt {attempt + 1}")
      return True

    except Exception as e:
      if attempt < max_retries - 1:
        logger.info(f"Error saving to GCS: {e}. Retrying in {retry_delay} seconds...")
        await asyncio.sleep(retry_delay)
      else:
        logger.info(f"Failed to save to GCS after {max_retries} attempts: {e}")
        return False


async def save_action_processing_time_to_gcs(action_processing_data: dict, user_id: str, stage_name: str, bucket_name: str):
  """Save action processing time data to Google Cloud Storage.
  
  Args:
      action_processing_data: Dictionary containing action processing timing information
      user_id: Unique identifier for the user
      stage_name: Name of the current stage
      bucket_name: Name of the GCS bucket
      
  Returns:
      bool: True if upload successful, False otherwise
  """
  try:
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Create filename with user and stage info in the "current" folder
    blob_filename = f"{BUCKET_DIR}/{timestamp}.json"
    
    # Prepare the data for upload
    upload_data = {
      "user_id": user_id,
      "stage_name": stage_name,
      "timestamp": datetime.now().isoformat(),
      "action_processing_data": action_processing_data
    }
    
    # Use the existing save_data_to_gcs function
    success = await save_data_to_gcs(upload_data, blob_filename, bucket_name)
    
    if success:
      logger.info(f"Successfully uploaded action processing time data for user {user_id}, stage {stage_name}")
    else:
      logger.warning(f"Failed to upload action processing time data for user {user_id}, stage {stage_name}")
    
    return success
    
  except Exception as e:
    logger.error(f"Error uploading action processing time data to GCS: {e}")
    return False


async def save_batch_action_processing_time_to_gcs(action_processing_records: list, user_id: str, stage_name: str, bucket_name: str):
  """Save multiple action processing time records in a batch to Google Cloud Storage.
  
  Args:
      action_processing_records: List of action processing time data dictionaries
      user_id: Unique identifier for the user
      stage_name: Name of the current stage
      bucket_name: Name of the GCS bucket
      
  Returns:
      bool: True if upload successful, False otherwise
  """
  try:
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Create filename in the "current" folder
    blob_filename = f"current/action_processing_time/{user_id}/{stage_name}/batch_{timestamp}.json"
    
    # Prepare the batch data
    upload_data = {
      "user_id": user_id,
      "stage_name": stage_name,
      "timestamp": datetime.now().isoformat(),
      "record_count": len(action_processing_records),
      "action_processing_records": action_processing_records
    }
    
    # Use the existing save_data_to_gcs function
    success = await save_data_to_gcs(upload_data, blob_filename, bucket_name)
    
    if success:
      logger.info(f"Successfully uploaded batch action processing time data for user {user_id}, stage {stage_name}")
    else:
      logger.warning(f"Failed to upload batch action processing time data for user {user_id}, stage {stage_name}")
    
    return success
    
  except Exception as e:
    logger.error(f"Error uploading batch action processing time data to GCS: {e}")
    return False
