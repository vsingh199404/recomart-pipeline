"""
Data Preparation and EDA Module for RecoMart Pipeline.
Cleans, encodes, normalizes data and generates exploratory analysis plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _clean_data(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Handle missing values: numerical → median, categorical → mode."""
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in df_clean.select_dtypes(include=['object']).columns:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])

    before = len(df)
    df_clean.dropna(inplace=True)
    after = len(df_clean)
    if before != after:
        logger.info(f"  {name}: Dropped {before - after} rows with remaining nulls")
    return df_clean


def _perform_eda(products: pd.DataFrame, interactions: pd.DataFrame, eda_dir: str):
    """Generate and save 6 EDA visualizations."""
    os.makedirs(eda_dir, exist_ok=True)
    logger.info(f"Generating EDA plots → {eda_dir}")

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        try:
            plt.style.use('ggplot')
        except Exception:
            pass

    # 1. Rating distribution
    num_cols = products.select_dtypes(include=[np.number]).columns
    rating_cols = [c for c in num_cols if 'rating' in c.lower()]
    if rating_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(products[rating_cols[0]].dropna(), bins=25, color='#3498db', edgecolor='white', alpha=0.8)
        ax.set_title('Product Rating Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, 'rating_distribution.png'), dpi=150)
        plt.close(fig)
        logger.info("  ✓ Rating distribution plot")

    # 2. Price distribution
    price_cols = [c for c in num_cols if 'price' in c.lower()]
    if price_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=products[price_cols[0]].dropna(), ax=ax, color='#2ecc71')
        ax.set_title('Price Distribution (Box Plot)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Price (₹)')
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, 'price_distribution.png'), dpi=150)
        plt.close(fig)
        logger.info("  ✓ Price distribution plot")

    # 3. Brand/Category frequency
    cat_cols = [c for c in products.columns if 'brand' in c.lower() or 'category' in c.lower()]
    if cat_cols:
        col = cat_cols[0]
        top_15 = products[col].value_counts().nlargest(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_15.values, y=top_15.index, ax=ax, palette='viridis')
        ax.set_title(f'Top 15 {col}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, 'category_brand_frequency.png'), dpi=150)
        plt.close(fig)
        logger.info("  ✓ Category/Brand frequency plot")

    # 4. Correlation heatmap
    num_df = products.select_dtypes(include=[np.number])
    if len(num_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(12, 9))
        corr = num_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax,
                    linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
        ax.set_title('Numerical Features Correlation Heatmap', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, 'correlation_heatmap.png'), dpi=150)
        plt.close(fig)
        logger.info("  ✓ Correlation heatmap")

    # 5. User activity distribution
    if 'user_id' in interactions.columns:
        user_counts = interactions['user_id'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(user_counts.values, bins=30, color='#e74c3c', edgecolor='white', alpha=0.8)
        ax.set_title('User Activity Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Interactions')
        ax.set_ylabel('Number of Users')
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, 'user_activity_distribution.png'), dpi=150)
        plt.close(fig)
        logger.info("  ✓ User activity distribution plot")

    # 6. Item popularity distribution
    if 'product_id' in interactions.columns:
        item_counts = interactions['product_id'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(item_counts.values, bins=30, color='#9b59b6', edgecolor='white', alpha=0.8)
        ax.set_title('Item Popularity Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Interactions')
        ax.set_ylabel('Number of Items')
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, 'item_popularity_distribution.png'), dpi=150)
        plt.close(fig)
        logger.info("  ✓ Item popularity distribution plot")


def _encode_and_scale(products: pd.DataFrame, interactions: pd.DataFrame, artifacts_dir: str):
    """Encode categorical features and scale numerical features."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # Encode categorical columns in products
    label_encoders = {}
    cat_keywords = ['gender', 'brand', 'holiday', 'season', 'geographical', 'location']
    for col in products.columns:
        if products[col].dtype == 'object' and any(kw in col.lower() for kw in cat_keywords):
            le = LabelEncoder()
            products[col] = le.fit_transform(products[col].astype(str))
            label_encoders[col] = le
            logger.info(f"  Encoded: {col} ({le.classes_.shape[0]} classes)")

    if label_encoders:
        joblib.dump(label_encoders, os.path.join(artifacts_dir, 'label_encoders.joblib'))

    # Scale numerical columns in products
    num_keywords = ['price', 'rating', 'sentiment', 'click', 'purchase']
    scale_cols = [c for c in products.select_dtypes(include=[np.number]).columns
                  if any(kw in c.lower() for kw in num_keywords)]

    if scale_cols:
        scaler = MinMaxScaler()
        products[scale_cols] = scaler.fit_transform(products[scale_cols])
        joblib.dump(scaler, os.path.join(artifacts_dir, 'products_scaler.joblib'))
        logger.info(f"  Scaled {len(scale_cols)} product columns: {scale_cols}")

    return products, interactions


def prepare_data(config: Dict[str, Any], validated_paths: Dict[str, str]) -> Dict[str, str]:
    """
    Main preparation function: clean, EDA, encode, normalize, save.

    Args:
        config: Pipeline config.
        validated_paths: Dict with 'product_data' and 'user_interactions' file paths.

    Returns:
        Dict with 'product_data' and 'user_interactions' paths to prepared files.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: DATA PREPARATION & EDA")
    logger.info("=" * 60)

    product_path = validated_paths.get('product_data')
    interaction_path = validated_paths.get('user_interactions')

    if not product_path or not os.path.exists(product_path):
        raise FileNotFoundError(f"Product data not found: {product_path}")
    if not interaction_path or not os.path.exists(interaction_path):
        raise FileNotFoundError(f"Interaction data not found: {interaction_path}")

    products = pd.read_csv(product_path)
    interactions = pd.read_csv(interaction_path)
    logger.info(f"Loaded: products={products.shape}, interactions={interactions.shape}")

    # Step 1: Clean
    logger.info("Cleaning data...")
    products = _clean_data(products, "Products")
    interactions = _clean_data(interactions, "Interactions")

    # Step 2: EDA
    eda_dir = config['paths'].get('eda_plots', os.path.join(config['paths']['reports'], 'eda_plots'))
    _perform_eda(products, interactions, eda_dir)

    # Step 3: Encode & Scale
    logger.info("Encoding and scaling...")
    prepared_dir = config['paths']['prepared']
    artifacts_dir = os.path.join(prepared_dir, 'artifacts')
    products, interactions = _encode_and_scale(products, interactions, artifacts_dir)

    # Step 4: Save
    os.makedirs(prepared_dir, exist_ok=True)
    out_products = os.path.join(prepared_dir, 'prepared_products.csv')
    out_interactions = os.path.join(prepared_dir, 'prepared_interactions.csv')

    products.to_csv(out_products, index=False)
    interactions.to_csv(out_interactions, index=False)

    logger.info(f"Prepared data saved → {prepared_dir}")

    return {
        'product_data': out_products,
        'user_interactions': out_interactions
    }
