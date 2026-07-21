# Task 3: Raw Data Storage — Structure Documentation

## Overview
Raw data ingested by the RecoMart pipeline is stored in a **local data lake** using a structured folder layout partitioned by **source**, **type**, and **timestamp (YYYY-MM-DD)**.

## Folder Structure

```
data/
├── raw/                              ← Raw ingested data (data lake)
│   ├── kaggle/                       ← Source: Kaggle dataset
│   │   └── product_recommendations/  ← Data type
│   │       └── YYYY-MM-DD/           ← Date partition
│   │           └── content_based_recommendation_dataset.csv
│   └── generated/                    ← Source: Simulated REST API
│       └── user_interactions/        ← Data type
│           └── YYYY-MM-DD/           ← Date partition
│               └── user_interactions.csv
│
├── validated/                        ← After validation stage
│   ├── validated_products.csv
│   └── validated_interactions.csv
│
├── prepared/                         ← After cleaning & encoding
│   ├── prepared_products.csv
│   ├── prepared_interactions.csv
│   └── artifacts/
│       ├── label_encoders.joblib
│       └── products_scaler.joblib
│
├── features/                         ← Engineered features
│   ├── product_features.csv
│   ├── product_features.parquet      ← Feast offline store source
│   ├── user_features.csv
│   ├── user_features.parquet         ← Feast offline store source
│   └── interaction_matrix.csv        ← User-Item matrix
│
├── versioning/                       ← Data lineage & versioning
│   ├── versions.json                 ← SHA-256 hash + metadata per file
│   └── lineage.json                  ← Transformation lineage graph
│
├── feast/                            ← Feast feature store
│   └── online_store.db               ← SQLite online store
│
└── warehouse.db                      ← SQLite data warehouse
```

## Partitioning Strategy

| Partition Level | Description | Example |
|---|---|---|
| **Source** | Origin of data | `kaggle/`, `generated/` |
| **Type** | Dataset category | `product_recommendations/`, `user_interactions/` |
| **Timestamp** | Ingestion date (ISO format) | `2026-07-19/` |

## Data Sources

| Source | Format | Ingestion Method | Update Frequency |
|---|---|---|---|
| Kaggle Ecommerce Dataset | CSV | `kagglehub.dataset_download()` | On pipeline run |
| Simulated REST API | CSV (generated) | Synthetic data generator | On pipeline run |

## Ingestion Scripts

| Script | Purpose |
|---|---|
| `src/ingestion/kaggle_ingestor.py` | Downloads Kaggle dataset with retry logic |
| `src/ingestion/api_ingestor.py` | Generates synthetic user interaction data |
| `src/ingestion/ingest_runner.py` | Unified runner calling both ingestors |

## Configuration
All paths are centrally managed in `config/pipeline_config.yaml` under the `paths:` section, using relative paths for portability across machines.
