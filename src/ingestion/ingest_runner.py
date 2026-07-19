"""Unified data ingestion runner for the RecoMart pipeline."""

from typing import Dict, Any
from src.utils.logger import get_logger
from src.ingestion.kaggle_ingestor import ingest_kaggle_data
from src.ingestion.api_ingestor import ingest_api_data

logger = get_logger(__name__)


def run_ingestion(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Runs the full ingestion pipeline: Kaggle dataset + simulated API data.

    Returns dict with keys:
        - 'product_data': path to the product CSV file
        - 'user_interactions': path to the user interactions CSV file
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA INGESTION")
    logger.info("=" * 60)

    results = {}

    # Step 1: Ingest Kaggle product dataset (CSV source)
    logger.info("[1/2] Ingesting Kaggle product dataset...")
    product_csv_path = ingest_kaggle_data(config)
    results['product_data'] = product_csv_path
    logger.info(f"  → Product data: {product_csv_path}")

    # Step 2: Generate synthetic user interactions (simulated REST API source)
    logger.info("[2/2] Generating user interaction data (simulated API)...")
    interactions_csv_path = ingest_api_data(config, product_data_path=product_csv_path)
    results['user_interactions'] = interactions_csv_path
    logger.info(f"  → User interactions: {interactions_csv_path}")

    logger.info("Data ingestion completed successfully.")
    return results
