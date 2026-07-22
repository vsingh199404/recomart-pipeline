"""Module for generating synthetic user interaction data, simulating a REST API."""

import os
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ingest_api_data(config: Dict[str, Any], product_data_path: Optional[str] = None) -> str:
    """
    Generates synthetic user interaction data and stores it as CSV.
    Simulates REST API ingestion with retry logic.

    Args:
        config: Pipeline configuration dictionary.
        product_data_path: Path to the product CSV file to infer product count.

    Returns:
        Path to the generated interactions CSV file.
    """
    raw_api_dir = config['paths']['raw_api']
    today_str = datetime.now().strftime('%Y-%m-%d')
    target_dir = os.path.join(raw_api_dir, today_str)
    os.makedirs(target_dir, exist_ok=True)

    target_file = os.path.join(target_dir, 'user_interactions.csv')
    max_retries = config.get('ingestion', {}).get('retry_attempts', 3)
    num_users = config.get('ingestion', {}).get('num_synthetic_users', 200)
    num_interactions = config.get('ingestion', {}).get('num_synthetic_interactions', 5000)

    # Determine number of products from the product dataset
    num_products = 500
    if product_data_path and os.path.exists(product_data_path):
        try:
            df_prod = pd.read_csv(product_data_path)
            num_products = len(df_prod)
            logger.info(f"Detected {num_products} products from {product_data_path}")
        except Exception as e:
            logger.warning(f"Could not read product data: {e}. Using default {num_products} products.")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Generating synthetic user interactions (Attempt {attempt}/{max_retries})...")
            logger.info(f"  Users: {num_users} | Interactions: {num_interactions} | Products: {num_products}")

            random.seed(42)
            np.random.seed(42)

            users = [f"U{str(i).zfill(3)}" for i in range(1, num_users + 1)]
            # Create skewed probability distributions for users and products
            user_probs = np.array([1.0 / (i + 1) for i in range(num_users)])
            user_probs /= user_probs.sum()
            
            prod_probs = np.array([1.0 / (i + 1) for i in range(num_products)])
            prod_probs /= prod_probs.sum()

            actions = ['view', 'click', 'purchase', 'rate']
            action_weights = [0.25, 0.25, 0.25, 0.25]  # Higher chance of rating/purchase

            end_time = datetime.now()
            start_time = end_time - timedelta(days=90)
            time_range = int((end_time - start_time).total_seconds())

            # Pre-select users and products using numpy for speed
            sampled_users = np.random.choice(users, size=num_interactions, p=user_probs)
            sampled_products = np.random.choice(num_products, size=num_interactions, p=prod_probs)
            sampled_actions = random.choices(actions, weights=action_weights, k=num_interactions)
            
            records = []
            for i in range(num_interactions):
                action = sampled_actions[i]

                # Rating only for 'rate' and 'purchase' actions
                rating = None
                if action == 'rate':
                    rating = round(random.uniform(1.0, 5.0), 1)
                elif action == 'purchase':
                    rating = round(random.uniform(3.0, 5.0), 1)

                timestamp = start_time + timedelta(seconds=random.randint(0, time_range))

                records.append({
                    'user_id': sampled_users[i],
                    'product_id': sampled_products[i],
                    'action_type': action,
                    'rating': rating,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })

            df = pd.DataFrame(records)
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Fill missing ratings with a default for views/clicks
            df['rating'] = df['rating'].fillna(0.0)

            df.to_csv(target_file, index=False)

            file_size = os.path.getsize(target_file)
            logger.info(f"Generated {num_interactions} interactions → {target_file} ({file_size} bytes)")
            return target_file

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed after {max_retries} attempts.")
                raise
