"""Module for ingesting datasets from Kaggle using kagglehub."""

import os
import shutil
import time
import glob
from datetime import datetime
from typing import Dict, Any

import kagglehub
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ingest_kaggle_data(config: Dict[str, Any]) -> str:
    """
    Downloads the Kaggle dataset using kagglehub and copies it to the raw data directory.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Path to the ingested CSV file.
    """
    dataset_name = config.get('ingestion', {}).get(
        'kaggle_dataset', 'kartikeybartwal/ecomerce-product-recommendation-dataset'
    )

    raw_kaggle_dir = config['paths']['raw_kaggle']
    today_str = datetime.now().strftime('%Y-%m-%d')
    target_dir = os.path.join(raw_kaggle_dir, today_str)
    os.makedirs(target_dir, exist_ok=True)

    max_retries = config.get('ingestion', {}).get('retry_attempts', 3)
    base_delay = config.get('ingestion', {}).get('retry_delay_seconds', 2)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading Kaggle dataset: {dataset_name} (Attempt {attempt}/{max_retries})")

            download_path = kagglehub.dataset_download(dataset_name)
            logger.info(f"Downloaded to: {download_path}")

            # Find all CSV files in the download path
            csv_files = glob.glob(os.path.join(download_path, "*.csv"))
            if not csv_files:
                # Check subdirectories
                csv_files = glob.glob(os.path.join(download_path, "**", "*.csv"), recursive=True)

            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {download_path}")

            # Copy CSV files to target directory
            ingested_file = None
            for src_file in csv_files:
                dst_file = os.path.join(target_dir, os.path.basename(src_file))
                shutil.copy2(src_file, dst_file)
                file_size = os.path.getsize(dst_file)

                # Read row count
                df = pd.read_csv(dst_file)
                row_count = len(df)

                logger.info(f"  Copied: {os.path.basename(src_file)} | Rows: {row_count} | Size: {file_size} bytes")
                ingested_file = dst_file  # Use the last (or only) CSV

            logger.info(f"Kaggle ingestion complete → {target_dir}")
            return ingested_file

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                sleep_time = base_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed after {max_retries} attempts.")
                raise
