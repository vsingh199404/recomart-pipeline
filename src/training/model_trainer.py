"""
Model Training and Evaluation Module for RecoMart Pipeline.
Trains Content-Based (GradientBoosting) and Collaborative Filtering (SVD) models.
Tracks experiments with MLflow.
"""

import os
import pickle
import random
import joblib
import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_ranking_metrics(predictions, k=10, threshold=3.5):
    """Compute Precision@K, Recall@K, and NDCG@K from Surprise predictions."""
    user_est_true = {}
    for uid, iid, true_r, est, _ in predictions:
        user_est_true.setdefault(uid, []).append((est, true_r))

    precisions, recalls, ndcgs = {}, {}, {}

    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(1 for _, tr in ratings if tr >= threshold)
        top_k = ratings[:k]
        n_rel_and_rec = sum(1 for est, tr in top_k if tr >= threshold and est >= threshold)
        n_rec = sum(1 for est, _ in top_k if est >= threshold)

        precisions[uid] = n_rel_and_rec / n_rec if n_rec > 0 else 0
        recalls[uid] = n_rel_and_rec / n_rel if n_rel > 0 else 0

        # NDCG@K
        dcg, idcg = 0, 0
        for i, (_, tr) in enumerate(top_k):
            rel = 1 if tr >= threshold else 0
            dcg += (2 ** rel - 1) / np.log2(i + 2)
        ideal = sorted(ratings, key=lambda x: x[1], reverse=True)[:k]
        for i, (_, tr) in enumerate(ideal):
            rel = 1 if tr >= threshold else 0
            idcg += (2 ** rel - 1) / np.log2(i + 2)
        ndcgs[uid] = dcg / idcg if idcg > 0 else 0

    return {
        'precision_at_k': np.mean(list(precisions.values())) if precisions else 0,
        'recall_at_k': np.mean(list(recalls.values())) if recalls else 0,
        'ndcg_at_k': np.mean(list(ndcgs.values())) if ndcgs else 0,
    }


def _train_content_based(config, feature_paths, model_dir):
    """Train a content-based model using product features."""
    logger.info("[Content-Based] Training GradientBoostingRegressor...")

    product_path = feature_paths.get('product_features')
    if not product_path or not os.path.exists(product_path):
        logger.warning(f"Product features not found: {product_path}")
        return None, {}

    df = pd.read_csv(product_path)

    # Find the target column (recommendation probability)
    target_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'probability' in col_lower or 'recommend' in col_lower:
            target_col = col
            break

    if target_col is None:
        # Use the last numerical column as target
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            target_col = num_cols[-1]
        else:
            logger.warning("No suitable target column found for content-based model.")
            return None, {}

    logger.info(f"  Target column: {target_col}")

    # Select numerical features (exclude target and id columns)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_col and 'id' not in c.lower() and 'timestamp' not in c.lower()]

    if len(feature_cols) == 0:
        logger.warning("No feature columns found.")
        return None, {}

    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)

    logger.info(f"  Features: {len(feature_cols)} columns, {len(X)} samples")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_cfg = config.get('model', {}).get('content_based', {})
    n_estimators = model_cfg.get('n_estimators', 100)
    max_depth = model_cfg.get('max_depth', 6)

    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'cb_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'cb_mae': float(mean_absolute_error(y_test, y_pred)),
        'cb_r2': float(r2_score(y_test, y_pred)),
    }

    logger.info(f"  Metrics: RMSE={metrics['cb_rmse']:.4f}, MAE={metrics['cb_mae']:.4f}, R2={metrics['cb_r2']:.4f}")

    with mlflow.start_run(run_name="content_based_model"):
        mlflow.log_param("algorithm", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("target_column", target_col)
        mlflow.log_metrics(metrics)

        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'content_based_model.joblib')
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

    logger.info(f"  Model saved: {model_path}")
    return model_path, metrics


def _train_collaborative_filtering(config, feature_paths, model_dir):
    """Train a collaborative filtering model using SVD."""
    logger.info("[Collaborative Filtering] Training SVD model...")

    # Use the interaction matrix (long format) or prepared interactions
    interaction_path = feature_paths.get('interaction_matrix') or feature_paths.get('user_interactions')
    if not interaction_path or not os.path.exists(interaction_path):
        logger.warning(f"Interaction data not found: {interaction_path}")
        return None, {}

    df = pd.read_csv(interaction_path)

    # Ensure we have user_id, product_id, rating
    required = ['user_id', 'product_id', 'rating']
    for col in required:
        if col not in df.columns:
            logger.warning(f"Missing column '{col}' in interaction data.")
            return None, {}

    # Filter to only rated interactions (rating > 0)
    df = df[df['rating'] > 0].copy()
    if len(df) < 10:
        logger.warning(f"Not enough rated interactions ({len(df)}) for collaborative filtering.")
        return None, {}

    logger.info(f"  Training data: {len(df)} rated interactions")

    try:
        from surprise import Dataset, Reader, SVD
        from surprise.model_selection import cross_validate
        from surprise import dump as surprise_dump

        reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
        data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

        model_cfg = config.get('model', {}).get('svd', {})
        n_factors = model_cfg.get('n_factors', 50)
        n_epochs = model_cfg.get('n_epochs', 20)

        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)

        # Cross-validation
        cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)

        metrics = {
            'cf_rmse': float(np.mean(cv_results['test_rmse'])),
            'cf_mae': float(np.mean(cv_results['test_mae'])),
        }

        # Train on full dataset
        trainset = data.build_full_trainset()
        algo.fit(trainset)

        # Ranking metrics on anti-testset
        testset = trainset.build_anti_testset()
        sample_size = min(len(testset), 5000)
        random.seed(42)
        test_sample = random.sample(testset, sample_size)
        predictions = algo.test(test_sample)

        ranking = _compute_ranking_metrics(predictions, k=10)
        metrics.update({f'cf_{k}': v for k, v in ranking.items()})

        logger.info(f"  Metrics: RMSE={metrics['cf_rmse']:.4f}, MAE={metrics['cf_mae']:.4f}")
        logger.info(f"  Ranking: P@10={ranking['precision_at_k']:.4f}, R@10={ranking['recall_at_k']:.4f}, NDCG@10={ranking['ndcg_at_k']:.4f}")

        with mlflow.start_run(run_name="collaborative_filtering_model"):
            mlflow.log_param("algorithm", "SVD")
            mlflow.log_param("n_factors", n_factors)
            mlflow.log_param("n_epochs", n_epochs)
            mlflow.log_param("n_interactions", len(df))
            mlflow.log_metrics(metrics)

            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'svd_model.pkl')
            try:
                surprise_dump.dump(model_path, algo=algo)
            except Exception:
                with open(model_path, 'wb') as f:
                    pickle.dump(algo, f)
            mlflow.log_artifact(model_path)

        logger.info(f"  Model saved: {model_path}")
        return model_path, metrics

    except ImportError as e:
        logger.error(f"Surprise library not available: {e}")
        return None, {}


def train_models(config, prepared_paths, feature_paths):
    """
    Train all recommendation models and log to MLflow.

    Args:
        config: Pipeline config.
        prepared_paths: Prepared data paths (for reference).
        feature_paths: Feature file paths from feature engineering.

    Returns:
        Dict with model paths and metrics.
    """
    logger.info("=" * 60)
    logger.info("STAGE 7: MODEL TRAINING & EVALUATION")
    logger.info("=" * 60)

    # Setup MLflow (using SQLite backend — required for MLflow 3.x+)
    mlflow_cfg = config.get('mlflow', {})
    mlflow_db_path = os.path.abspath('mlflow.db')
    tracking_uri = f"sqlite:///{mlflow_db_path}"

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_cfg.get('experiment_name', 'recomart_recommendations')
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow URI: {tracking_uri}")
    logger.info(f"Experiment: {experiment_name}")

    model_dir = config['paths'].get('models', 'models')
    os.makedirs(model_dir, exist_ok=True)

    results = {'model_paths': {}, 'metrics': {}}

    # 1. Content-based model
    cb_path, cb_metrics = _train_content_based(config, feature_paths, model_dir)
    if cb_path:
        results['model_paths']['content_based'] = cb_path
        results['metrics']['content_based'] = cb_metrics

    # 2. Collaborative filtering model
    cf_path, cf_metrics = _train_collaborative_filtering(config, feature_paths, model_dir)
    if cf_path:
        results['model_paths']['collaborative_filtering'] = cf_path
        results['metrics']['collaborative_filtering'] = cf_metrics

    logger.info("=" * 60)
    logger.info("MODEL TRAINING COMPLETE")
    for model_name, metrics in results['metrics'].items():
        logger.info(f"  {model_name}: {metrics}")
    logger.info("=" * 60)

    return results
