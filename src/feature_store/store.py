import os
from datetime import datetime
from feast import FeatureStore
from typing import Dict, Any, List, Optional
import pandas as pd
from src.utils.logger import get_logger
from src.feature_store.feature_definitions import (
    product_entity, user_entity, product_features_view, user_features_view
)

logger = get_logger(__name__)

PROJECT_ROOT = "i:/projects/recomart-pipeline"
FEAST_REPO_PATH = os.path.join(PROJECT_ROOT, "src", "feature_store")
FEAST_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "feast")

def setup_feast(config: Dict[str, Any]) -> Optional[FeatureStore]:
    """Initialize Feast FeatureStore and apply definitions."""
    try:
        os.makedirs(FEAST_DATA_DIR, exist_ok=True)
        store = FeatureStore(repo_path=FEAST_REPO_PATH)
        
        logger.info("Applying Feast definitions...")
        store.apply([
            product_entity, 
            user_entity, 
            product_features_view, 
            user_features_view
        ])
        logger.info("Successfully registered features with Feast.")
        return store
    except Exception as e:
        logger.warning(f"Failed to setup Feast: {e}")
        return None

def materialize_features(config: Dict[str, Any]) -> None:
    """Materialize features to the online store."""
    try:
        store = FeatureStore(repo_path=FEAST_REPO_PATH)
        logger.info("Materializing features to online store...")
        store.materialize_incremental(end_date=datetime.now())
        logger.info("Successfully materialized features.")
    except Exception as e:
        logger.warning(f"Failed to materialize features: {e}")

def get_training_data(config: Dict[str, Any], entity_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Retrieve historical features for training."""
    try:
        store = FeatureStore(repo_path=FEAST_REPO_PATH)
        logger.info("Retrieving historical features...")
        features = [
            "product_features:click_to_purchase_ratio",
            "product_features:price_bracket",
            "product_features:brand_popularity",
            "product_features:quality_score",
            "product_features:rating",
            "product_features:sentiment_score",
            "product_features:price",
            "user_features:activity_frequency",
            "user_features:avg_rating_given",
            "user_features:rating_variance",
            "user_features:unique_items_interacted",
            "user_features:purchase_ratio"
        ]
        job = store.get_historical_features(
            entity_df=entity_df,
            features=features
        )
        return job.to_df()
    except Exception as e:
        logger.warning(f"Failed to get historical features: {e}")
        return None

def get_online_features(config: Dict[str, Any], entity_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Retrieve online features for inference."""
    try:
        store = FeatureStore(repo_path=FEAST_REPO_PATH)
        features = [
            "product_features:click_to_purchase_ratio",
            "product_features:price_bracket",
            "product_features:brand_popularity",
            "product_features:quality_score",
            "product_features:rating",
            "product_features:sentiment_score",
            "product_features:price",
            "user_features:activity_frequency",
            "user_features:avg_rating_given",
            "user_features:rating_variance",
            "user_features:unique_items_interacted",
            "user_features:purchase_ratio"
        ]
        response = store.get_online_features(
            features=features,
            entity_rows=entity_rows
        )
        return response.to_dict()
    except Exception as e:
        logger.warning(f"Failed to get online features: {e}")
        return None

def run_feature_store(config: Dict[str, Any], feature_paths: Dict[str, str]) -> Dict[str, Any]:
    """Setup and materialize the feature store."""
    logger.info("Starting feature store operations...")
    store = setup_feast(config)
    
    if store is not None:
        materialize_features(config)
        return {"status": "success", "repo_path": FEAST_REPO_PATH}
    else:
        return {"status": "failed", "repo_path": FEAST_REPO_PATH}
