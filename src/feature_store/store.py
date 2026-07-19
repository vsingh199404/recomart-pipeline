"""
Feast Feature Store operations for the RecoMart pipeline.
Configured programmatically to avoid Windows path resolution issues.
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEAST_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'feast')
FEAST_REPO_PATH = os.path.join(PROJECT_ROOT, 'src', 'feature_store')


def _get_repo_config():
    """Build a Feast RepoConfig programmatically.
    
    Sets repo_path=FEAST_DATA_DIR so that simple relative filenames like
    'registry.db' resolve to the correct absolute location without Feast
    misinterpreting the Windows drive letter as a URL scheme.
    """
    from pathlib import Path
    from feast import RepoConfig
    from feast.infra.registry.registry import RegistryConfig
    from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig

    os.makedirs(FEAST_DATA_DIR, exist_ok=True)

    return RepoConfig(
        project='recomart',
        provider='local',
        repo_path=Path(FEAST_DATA_DIR),         # Feast resolves relative paths from here
        registry=RegistryConfig(path='registry.db'),          # → FEAST_DATA_DIR/registry.db
        online_store=SqliteOnlineStoreConfig(path='online_store.db'),  # → FEAST_DATA_DIR/online_store.db
        entity_key_serialization_version=3,
    )


def setup_feast(config: Dict[str, Any]) -> Optional[Any]:
    """Initialize Feast FeatureStore and apply feature definitions."""
    try:
        from feast import FeatureStore
        from src.feature_store.feature_definitions import (
            product_entity, user_entity,
            product_features_view, user_features_view
        )

        os.makedirs(FEAST_DATA_DIR, exist_ok=True)

        repo_config = _get_repo_config()
        store = FeatureStore(config=repo_config)

        logger.info("Applying Feast feature definitions...")
        store.apply([
            product_entity,
            user_entity,
            product_features_view,
            user_features_view,
        ])
        logger.info("Successfully registered features with Feast.")
        return store

    except Exception as e:
        logger.warning(f"Failed to setup Feast: {e}")
        return None


def materialize_features(config: Dict[str, Any]) -> None:
    """Materialize features from offline to online store."""
    try:
        from feast import FeatureStore
        repo_config = _get_repo_config()
        store = FeatureStore(config=repo_config)

        logger.info("Materializing features to online store...")
        store.materialize_incremental(end_date=datetime.now(tz=timezone.utc))
        logger.info("Successfully materialized features.")

    except Exception as e:
        logger.warning(f"Failed to materialize features: {e}")


def get_training_data(config: Dict[str, Any], entity_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Retrieve historical features for model training."""
    try:
        from feast import FeatureStore
        repo_config = _get_repo_config()
        store = FeatureStore(config=repo_config)

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
            "user_features:purchase_ratio",
        ]
        job = store.get_historical_features(entity_df=entity_df, features=features)
        return job.to_df()

    except Exception as e:
        logger.warning(f"Failed to get historical features: {e}")
        return None


def get_online_features(config: Dict[str, Any], entity_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Retrieve online features for real-time inference."""
    try:
        from feast import FeatureStore
        repo_config = _get_repo_config()
        store = FeatureStore(config=repo_config)

        features = [
            "product_features:click_to_purchase_ratio",
            "product_features:brand_popularity",
            "product_features:quality_score",
            "user_features:activity_frequency",
            "user_features:avg_rating_given",
            "user_features:purchase_ratio",
        ]
        response = store.get_online_features(features=features, entity_rows=entity_rows)
        return response.to_dict()

    except Exception as e:
        logger.warning(f"Failed to get online features: {e}")
        return None


def run_feature_store(config: Dict[str, Any], feature_paths: Dict[str, str]) -> Dict[str, Any]:
    """Main entry point: setup Feast and materialize features."""
    logger.info("Starting feature store operations...")

    store = setup_feast(config)
    if store is not None:
        materialize_features(config)
        logger.info(f"Feature store ready. Data dir: {FEAST_DATA_DIR}")
        return {"status": "success", "feast_data_dir": FEAST_DATA_DIR}
    else:
        return {"status": "failed", "feast_data_dir": FEAST_DATA_DIR}
