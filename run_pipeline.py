"""
RecoMart Pipeline — Main Entry Point
Runs the full end-to-end recommendation pipeline via Prefect.
"""

import sys
import os

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Change working directory to project root (important for relative paths in config)
os.chdir(PROJECT_ROOT)

import logging
from src.utils.logger import get_logger

# Ensure Prefect logs are also captured in pipline.logs
log_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
prefect_log_file = os.path.join(log_dir, "pipline.logs")

prefect_logger = logging.getLogger("prefect")
if not any(isinstance(h, logging.FileHandler) and "pipline.logs" in getattr(h, 'baseFilename', '') for h in prefect_logger.handlers):
    file_handler = logging.FileHandler(prefect_log_file, mode="a", encoding="utf-8")
    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    prefect_logger.addHandler(file_handler)

logger = get_logger("run_pipeline")


def main():
    logger.info("=" * 70)
    logger.info("  RecoMart — End-to-End Data Management Pipeline")
    logger.info("  Project Root: %s", PROJECT_ROOT)
    logger.info("=" * 70)

    try:
        from orchestration.pipeline_flow import recomart_pipeline

        results = recomart_pipeline()

        logger.info("")
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("")

        if results and 'metrics' in results:
            for model_name, metrics in results['metrics'].items():
                logger.info(f"  📊 {model_name}:")
                for k, v in metrics.items():
                    logger.info(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")

        return results

    except Exception as e:
        logger.exception(f"❌ Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
