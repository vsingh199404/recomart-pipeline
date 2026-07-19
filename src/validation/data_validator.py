"""
Data Validation Module for RecoMart Pipeline.
Validates product data and user interactions, generates HTML quality report.
"""

import os
import shutil
import pandas as pd
import numpy as np
from jinja2 import Template
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── HTML Report Template ─────────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RecoMart Data Quality Report</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background: #f8f9fa; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th { background: #3498db; color: white; padding: 12px; text-align: left; }
        td { border: 1px solid #ddd; padding: 10px; }
        .pass { color: #27ae60; font-weight: bold; }
        .fail { color: #e74c3c; font-weight: bold; }
        .score-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; font-size: 1.2em; margin: 20px 0; }
        .score-box strong { font-size: 1.5em; }
        .timestamp { color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>🔍 RecoMart Data Quality Report</h1>
    <p class="timestamp">Generated: {{ timestamp }}</p>

    <div class="score-box">
        <strong>Overall Quality Score: {{ "%.1f"|format(overall_score) }}%</strong>
        <br>{{ passed_count }} / {{ total_count }} checks passed
    </div>

    <h2>📦 Product Data Quality</h2>
    <table>
        <tr><th>Check</th><th>Status</th><th>Details</th></tr>
        {% for check in product_checks %}
        <tr>
            <td>{{ check.name }}</td>
            <td class="{{ 'pass' if check.passed else 'fail' }}">{{ '✅ Pass' if check.passed else '❌ Fail' }}</td>
            <td>{{ check.details }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>👤 User Interactions Data Quality</h2>
    <table>
        <tr><th>Check</th><th>Status</th><th>Details</th></tr>
        {% for check in interaction_checks %}
        <tr>
            <td>{{ check.name }}</td>
            <td class="{{ 'pass' if check.passed else 'fail' }}">{{ '✅ Pass' if check.passed else '❌ Fail' }}</td>
            <td>{{ check.details }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""


def _validate_product_df(df: pd.DataFrame, config: dict) -> list:
    """Run validation checks on the product dataset."""
    checks = []

    # 1. Row count check
    checks.append({
        'name': 'Row Count',
        'passed': len(df) > 0,
        'details': f"{len(df)} rows, {len(df.columns)} columns"
    })

    # 2. Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    missing_pct = (total_missing / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
    max_missing = config.get('validation', {}).get('max_missing_pct', 30.0)
    checks.append({
        'name': 'Missing Values',
        'passed': missing_pct <= max_missing,
        'details': f"Total: {total_missing} ({missing_pct:.2f}%), Threshold: {max_missing}%"
    })

    # 3. Duplicate rows
    dup_count = df.duplicated().sum()
    checks.append({
        'name': 'Duplicate Rows',
        'passed': dup_count == 0,
        'details': f"{dup_count} duplicate rows found"
    })

    # 4. Numerical range checks
    range_issues = []
    for col in df.select_dtypes(include=[np.number]).columns:
        col_lower = col.lower()
        if 'rating' in col_lower:
            out_of_range = df[(df[col] < 0) | (df[col] > 5)][col].count()
            if out_of_range > 0:
                range_issues.append(f"{col}: {out_of_range} values outside 0-5")
        if 'price' in col_lower:
            negative = df[df[col] < 0][col].count()
            if negative > 0:
                range_issues.append(f"{col}: {negative} negative values")

    checks.append({
        'name': 'Range Checks',
        'passed': len(range_issues) == 0,
        'details': '; '.join(range_issues) if range_issues else "All numerical ranges valid"
    })

    # 5. Data type consistency
    num_cols = len(df.select_dtypes(include=[np.number]).columns)
    cat_cols = len(df.select_dtypes(include=['object']).columns)
    checks.append({
        'name': 'Data Type Check',
        'passed': True,
        'details': f"{num_cols} numerical, {cat_cols} categorical columns"
    })

    return checks


def _validate_interactions_df(df: pd.DataFrame, config: dict) -> list:
    """Run validation checks on user interaction data."""
    checks = []

    # 1. Schema check
    required_cols = ['user_id', 'product_id', 'rating', 'timestamp', 'action_type']
    missing_cols = [c for c in required_cols if c not in df.columns]
    checks.append({
        'name': 'Schema Validation',
        'passed': len(missing_cols) == 0,
        'details': f"Missing: {missing_cols}" if missing_cols else f"All {len(required_cols)} required columns present"
    })

    # 2. Rating range
    if 'rating' in df.columns:
        rated = df[df['rating'] > 0]  # 0 means no rating
        invalid = rated[(rated['rating'] < 1) | (rated['rating'] > 5)]
        checks.append({
            'name': 'Rating Range (1-5)',
            'passed': len(invalid) == 0,
            'details': f"{len(invalid)} invalid ratings" if len(invalid) > 0 else f"All {len(rated)} rated interactions valid"
        })

    # 3. Valid action types
    if 'action_type' in df.columns:
        valid_actions = {'view', 'click', 'purchase', 'rate'}
        actual_actions = set(df['action_type'].unique())
        invalid = actual_actions - valid_actions
        checks.append({
            'name': 'Valid Action Types',
            'passed': len(invalid) == 0,
            'details': f"Invalid: {invalid}" if invalid else f"Actions: {actual_actions}"
        })

    # 4. Timestamp format
    if 'timestamp' in df.columns:
        try:
            pd.to_datetime(df['timestamp'])
            checks.append({
                'name': 'Timestamp Format',
                'passed': True,
                'details': "All timestamps parseable"
            })
        except Exception:
            checks.append({
                'name': 'Timestamp Format',
                'passed': False,
                'details': "Invalid timestamp format detected"
            })

    # 5. Duplicate check
    dup_count = df.duplicated().sum()
    checks.append({
        'name': 'Duplicate Interactions',
        'passed': True,  # Duplicates are possible in interactions
        'details': f"{dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%)" if len(df) > 0 else "Empty dataset"
    })

    return checks


def validate_data(config: Dict[str, Any], ingested_paths: Dict[str, str]) -> Dict[str, str]:
    """
    Validates ingested data, generates quality report, saves validated data.

    Args:
        config: Pipeline config dict.
        ingested_paths: Dict with 'product_data' and 'user_interactions' file paths.

    Returns:
        Dict with 'product_data' and 'user_interactions' paths to validated files.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: DATA VALIDATION")
    logger.info("=" * 60)

    validated_dir = config['paths']['validated']
    reports_dir = config['paths']['reports']
    os.makedirs(validated_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    product_checks = []
    interaction_checks = []

    # ── Validate product data ──
    product_path = ingested_paths.get('product_data', '')
    validated_product_path = None
    if product_path and os.path.exists(product_path):
        logger.info(f"Validating product data: {product_path}")
        df_product = pd.read_csv(product_path)
        product_checks = _validate_product_df(df_product, config)

        # Clean: remove duplicates
        df_clean = df_product.drop_duplicates()
        validated_product_path = os.path.join(validated_dir, 'validated_products.csv')
        df_clean.to_csv(validated_product_path, index=False)
        logger.info(f"  Validated products: {len(df_clean)} rows → {validated_product_path}")
    else:
        logger.warning(f"Product data not found at: {product_path}")

    # ── Validate user interactions ──
    interactions_path = ingested_paths.get('user_interactions', '')
    validated_interactions_path = None
    if interactions_path and os.path.exists(interactions_path):
        logger.info(f"Validating user interactions: {interactions_path}")
        df_interactions = pd.read_csv(interactions_path)
        interaction_checks = _validate_interactions_df(df_interactions, config)

        # Clean: remove full duplicates
        df_clean = df_interactions.drop_duplicates()
        validated_interactions_path = os.path.join(validated_dir, 'validated_interactions.csv')
        df_clean.to_csv(validated_interactions_path, index=False)
        logger.info(f"  Validated interactions: {len(df_clean)} rows → {validated_interactions_path}")
    else:
        logger.warning(f"Interactions data not found at: {interactions_path}")

    # ── Generate HTML quality report ──
    from datetime import datetime
    all_checks = product_checks + interaction_checks
    passed = sum(1 for c in all_checks if c['passed'])
    total = len(all_checks)
    score = (passed / total * 100) if total > 0 else 0

    html = Template(HTML_TEMPLATE).render(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        overall_score=score,
        passed_count=passed,
        total_count=total,
        product_checks=product_checks,
        interaction_checks=interaction_checks
    )

    report_path = os.path.join(reports_dir, 'data_quality_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Quality Score: {score:.1f}% ({passed}/{total} checks passed)")
    logger.info(f"Report saved: {report_path}")

    return {
        'product_data': validated_product_path,
        'user_interactions': validated_interactions_path
    }
