"""
Feature Engineering Module for RecoMart Pipeline.
Creates product features, user features, and user-item interaction matrix.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


def engineer_features(config: dict, prepared_paths: Dict[str, str]) -> Dict[str, str]:
    """
    Generate engineered features from prepared data.

    Args:
        config: Pipeline config.
        prepared_paths: Dict with 'product_data' and 'user_interactions' file paths.

    Returns:
        Dict with paths to all generated feature files.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: FEATURE ENGINEERING")
    logger.info("=" * 60)

    product_path = prepared_paths.get('product_data')
    interaction_path = prepared_paths.get('user_interactions')

    if not product_path or not os.path.exists(product_path):
        raise FileNotFoundError(f"Product data not found: {product_path}")
    if not interaction_path or not os.path.exists(interaction_path):
        raise FileNotFoundError(f"Interaction data not found: {interaction_path}")

    products = pd.read_csv(product_path)
    interactions = pd.read_csv(interaction_path)
    logger.info(f"Loaded: products={products.shape}, interactions={interactions.shape}")

    feature_dir = config['paths']['features']
    os.makedirs(feature_dir, exist_ok=True)

    # ── 1. Product Features ──
    logger.info("Engineering product features...")
    product_features = _engineer_product_features(products)

    # ── 2. User Features ──
    logger.info("Engineering user features...")
    user_features = _engineer_user_features(interactions)

    # ── 3. User-Item Interaction Matrix ──
    logger.info("Building user-item interaction matrix...")
    interaction_matrix = _build_interaction_matrix(interactions)

    # ── Save outputs ──
    paths = {}

    # Product features
    prod_csv = os.path.join(feature_dir, 'product_features.csv')
    prod_parquet = os.path.join(feature_dir, 'product_features.parquet')
    _save_features(product_features, 'product_id', prod_csv, prod_parquet)
    paths['product_features'] = prod_csv
    paths['product_features_parquet'] = prod_parquet

    # User features
    user_csv = os.path.join(feature_dir, 'user_features.csv')
    user_parquet = os.path.join(feature_dir, 'user_features.parquet')
    _save_features(user_features, 'user_id', user_csv, user_parquet)
    paths['user_features'] = user_csv
    paths['user_features_parquet'] = user_parquet

    # Interaction matrix
    matrix_csv = os.path.join(feature_dir, 'interaction_matrix.csv')
    interaction_matrix.to_csv(matrix_csv, index=False)
    paths['interaction_matrix'] = matrix_csv

    # Also pass through the prepared interactions (needed for model training)
    paths['user_interactions'] = interaction_path

    # ── Save to SQLite warehouse ──
    _save_to_sqlite(product_features, user_features, config)

    logger.info(f"Feature engineering complete. {len(paths)} files saved → {feature_dir}")
    return paths


def _engineer_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create product-level features."""
    features = df.copy()

    # Assign product_id if not present
    if 'product_id' not in features.columns:
        features.insert(0, 'product_id', range(len(features)))
    features['product_id'] = features['product_id'].astype(int)

    # Identify columns by keyword matching (dataset may have varying names)
    cols = {c.lower(): c for c in features.columns}

    # click_to_purchase_ratio
    click_col = next((cols[k] for k in cols if 'click' in k and 'similar' in k), None)
    purch_col = next((cols[k] for k in cols if 'purchase' in k and 'similar' in k), None)
    if click_col and purch_col:
        features['click_to_purchase_ratio'] = features[click_col] / (features[purch_col] + 1)
    else:
        features['click_to_purchase_ratio'] = 0.0

    # price_bracket
    price_col = next((cols[k] for k in cols if 'price' in k and 'median' not in k), None)
    if not price_col:
        price_col = next((cols[k] for k in cols if 'price' in k), None)
    if price_col:
        try:
            features['price_bracket'] = pd.qcut(
                features[price_col], q=4,
                labels=[0, 1, 2, 3],  # 0=Low, 1=Medium, 2=High, 3=Premium
                duplicates='drop'
            ).astype(int)
        except Exception:
            features['price_bracket'] = 0

    # brand_popularity
    brand_col = next((cols[k] for k in cols if 'brand' in k), None)
    if brand_col:
        brand_counts = features[brand_col].value_counts(normalize=True)
        features['brand_popularity'] = features[brand_col].map(brand_counts).fillna(0)
    else:
        features['brand_popularity'] = 0.0

    # quality_score
    rating_col = next((cols[k] for k in cols if 'rating' in k and 'similar' not in k and 'avg' not in k.lower()), None)
    sentiment_col = next((cols[k] for k in cols if 'sentiment' in k), None)
    if rating_col and sentiment_col:
        features['quality_score'] = features[rating_col] * 0.7 + features[sentiment_col] * 0.3
    elif rating_col:
        features['quality_score'] = features[rating_col]
    else:
        features['quality_score'] = 0.0

    # seasonal_demand
    season_col = next((cols[k] for k in cols if 'season' in k), None)
    holiday_col = next((cols[k] for k in cols if 'holiday' in k), None)
    if season_col is not None and holiday_col is not None:
        features['seasonal_demand'] = features[season_col].astype(str) + '_' + features[holiday_col].astype(str)
    else:
        features['seasonal_demand'] = '0_0'

    logger.info(f"  Product features: {features.shape} ({len(features.columns)} columns)")
    return features


def _engineer_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create user-level features from interactions."""
    df_copy = df.copy()

    # Convert user_id to numeric for grouping
    if df_copy['user_id'].dtype == 'object':
        # User IDs like U001 → extract numeric part
        df_copy['user_id_num'] = df_copy['user_id'].str.extract(r'(\d+)').astype(int)
    else:
        df_copy['user_id_num'] = df_copy['user_id'].astype(int)

    grp = df_copy.groupby('user_id_num')

    features = pd.DataFrame()
    features['user_id'] = sorted(df_copy['user_id_num'].unique())
    features = features.set_index('user_id')

    # activity_frequency
    features['activity_frequency'] = grp.size()

    # avg_rating_given (only rated interactions, rating > 0)
    rated = df_copy[df_copy['rating'] > 0]
    if len(rated) > 0:
        rated_grp = rated.groupby('user_id_num')
        features['avg_rating_given'] = rated_grp['rating'].mean()
        features['rating_variance'] = rated_grp['rating'].std().fillna(0)
    else:
        features['avg_rating_given'] = 0.0
        features['rating_variance'] = 0.0

    features['avg_rating_given'] = features['avg_rating_given'].fillna(0)
    features['rating_variance'] = features['rating_variance'].fillna(0)

    # unique_items_interacted
    features['unique_items_interacted'] = grp['product_id'].nunique()

    # purchase_ratio
    if 'action_type' in df_copy.columns:
        purchases = df_copy[df_copy['action_type'] == 'purchase'].groupby('user_id_num').size()
        features['purchase_ratio'] = (purchases / features['activity_frequency']).fillna(0)
    else:
        features['purchase_ratio'] = 0.0

    features = features.reset_index()
    features = features.rename(columns={'index': 'user_id'})
    if 'user_id' not in features.columns:
        features = features.rename(columns={'user_id_num': 'user_id'})

    features['user_id'] = features['user_id'].astype(int)

    logger.info(f"  User features: {features.shape} ({len(features)} users)")
    return features


def _build_interaction_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build user-item interaction matrix (long format for model training)."""
    df_copy = df.copy()

    if df_copy['user_id'].dtype == 'object':
        df_copy['user_id'] = df_copy['user_id'].str.extract(r'(\d+)').astype(int)

    # Keep only rated interactions
    rated = df_copy[df_copy['rating'] > 0][['user_id', 'product_id', 'rating']].copy()

    # Average rating per user-product pair
    if len(rated) > 0:
        matrix = rated.groupby(['user_id', 'product_id'])['rating'].mean().reset_index()
    else:
        matrix = pd.DataFrame(columns=['user_id', 'product_id', 'rating'])

    logger.info(f"  Interaction matrix: {len(matrix)} user-product pairs")
    return matrix


def _save_features(df: pd.DataFrame, id_col: str, csv_path: str, parquet_path: str):
    """Save features to CSV and Parquet (with Feast-required event_timestamp)."""
    df.to_csv(csv_path, index=False)

    df_pq = df.copy()
    df_pq['event_timestamp'] = pd.Timestamp.now(tz='UTC')

    # Convert object columns to string for Parquet compatibility
    for col in df_pq.select_dtypes(include=['object']).columns:
        df_pq[col] = df_pq[col].astype(str)

    df_pq.to_parquet(parquet_path, engine='pyarrow', index=False)
    logger.info(f"  Saved: {os.path.basename(csv_path)} + {os.path.basename(parquet_path)}")


def _save_to_sqlite(product_df: pd.DataFrame, user_df: pd.DataFrame, config: dict):
    """Save features to SQLite data warehouse."""
    db_path = config['paths'].get('warehouse', 'data/warehouse.db')
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)

    # Convert object columns for SQLite compatibility
    for df in [product_df, user_df]:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)

    with sqlite3.connect(db_path) as conn:
        product_df.to_sql('product_features', conn, if_exists='replace', index=False)
        user_df.to_sql('user_features', conn, if_exists='replace', index=False)

    logger.info(f"  Features saved to SQLite: {db_path}")
