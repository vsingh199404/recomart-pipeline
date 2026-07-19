"""
Prefect Pipeline Orchestration for RecoMart.
Defines the full DAG: ingestion → validation → preparation → features → feast → versioning → training.
"""

import os
import sys
import yaml
from prefect import flow, task, get_run_logger

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.ingest_runner import run_ingestion
from src.validation.data_validator import validate_data
from src.preparation.data_preparer import prepare_data
from src.transformation.feature_engineer import engineer_features
from src.feature_store.store import run_feature_store
from src.versioning.data_versioner import version_data
from src.training.model_trainer import train_models


def load_config() -> dict:
    """Load pipeline configuration from YAML."""
    config_path = os.path.join(PROJECT_ROOT, 'config', 'pipeline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ── Prefect Tasks ──────────────────────────────────────────────────────

@task(name="Data Ingestion", retries=2, retry_delay_seconds=10)
def task_ingest(config: dict) -> dict:
    logger = get_run_logger()
    logger.info("🚀 Starting Data Ingestion")
    result = run_ingestion(config)
    logger.info(f"✅ Ingestion complete: {list(result.keys())}")
    return result


@task(name="Data Validation")
def task_validate(config: dict, ingested: dict) -> dict:
    logger = get_run_logger()
    logger.info("🔍 Starting Data Validation")
    result = validate_data(config, ingested)
    logger.info("✅ Validation complete")
    return result


@task(name="Data Preparation")
def task_prepare(config: dict, validated: dict) -> dict:
    logger = get_run_logger()
    logger.info("🧹 Starting Data Preparation & EDA")
    result = prepare_data(config, validated)
    logger.info("✅ Preparation complete")
    return result


@task(name="Feature Engineering")
def task_engineer(config: dict, prepared: dict) -> dict:
    logger = get_run_logger()
    logger.info("⚙️ Starting Feature Engineering")
    result = engineer_features(config, prepared)
    logger.info(f"✅ Feature Engineering complete: {len(result)} outputs")
    return result


@task(name="Feature Store (Feast)")
def task_feature_store(config: dict, features: dict) -> dict:
    logger = get_run_logger()
    logger.info("📦 Starting Feature Store (Feast)")
    result = run_feature_store(config, features)
    logger.info(f"✅ Feature Store: {result.get('status', 'unknown')}")
    return result


@task(name="Data Versioning")
def task_version(config: dict, features: dict) -> dict:
    logger = get_run_logger()
    logger.info("📋 Starting Data Versioning")
    result = version_data(config, features)
    logger.info(f"✅ Versioning complete: v{result.get('version_number', '?')}")
    return result


@task(name="Model Training")
def task_train(config: dict, prepared: dict, features: dict) -> dict:
    logger = get_run_logger()
    logger.info("🧠 Starting Model Training")
    result = train_models(config, prepared, features)
    logger.info("✅ Model Training complete")
    return result


# ── Prefect Flow ───────────────────────────────────────────────────────

@flow(name="RecoMart Recommendation Pipeline", log_prints=True)
def recomart_pipeline():
    """
    Full end-to-end pipeline flow:
    Ingestion → Validation → Preparation → Feature Engineering
    → Feature Store → Data Versioning → Model Training
    """
    print("=" * 70)
    print("  RecoMart Recommendation Pipeline — Starting Execution")
    print("=" * 70)

    config = load_config()

    # Step 1: Data Ingestion (2 sources: Kaggle CSV + Simulated API)
    ingested = task_ingest(config)

    # Step 2: Data Validation & Quality Report
    validated = task_validate(config, ingested)

    # Step 3: Data Preparation & EDA
    prepared = task_prepare(config, validated)

    # Step 4: Feature Engineering
    features = task_engineer(config, prepared)

    # Step 5: Feature Store (Feast) — register & materialize
    task_feature_store(config, features)

    # Step 6: Data Versioning & Lineage
    task_version(config, features)

    # Step 7: Model Training & Evaluation (MLflow)
    results = task_train(config, prepared, features)

    print("=" * 70)
    print("  Pipeline Execution Complete!")
    print("=" * 70)

    return results
