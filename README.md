# RecoMart: End-to-End Data Management Pipeline for Recommendation System

## 📋 Project Overview

RecoMart is a **data-driven recommendation engine** for an e-commerce platform. This project implements a complete, automated, modular data pipeline that supports:

- **Batch and near-real-time data ingestion** from multiple sources
- **Data validation, cleaning, and preparation** with quality reporting
- **Feature engineering** with both user and product features
- **Feature store management** using Feast
- **Data versioning and lineage tracking**
- **Model training** (Collaborative Filtering + Content-Based)
- **Experiment tracking** with MLflow
- **Pipeline orchestration** with Prefect

---

## 🏗️ Architecture

```
┌──────────────────┐     ┌──────────────────┐
│   Kaggle CSV     │     │  Simulated API   │
│  (Product Data)  │     │(User Interactions)│
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
         ┌──────────────────┐
         │   Data Ingestion  │  (Stage 1)
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │  Data Validation  │  (Stage 2) → HTML Quality Report
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ Data Preparation  │  (Stage 3) → EDA Plots
         │   + EDA           │
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │    Feature        │  (Stage 4) → SQLite Warehouse
         │   Engineering     │
         └────────┬─────────┘
                  ▼
    ┌─────────────┼──────────────┐
    ▼             ▼              ▼
┌────────┐ ┌──────────┐  ┌──────────┐
│  Feast │ │Versioning│  │  Model   │
│Feature │ │& Lineage │  │ Training │  (Stage 5-7)
│ Store  │ │          │  │ (MLflow) │
└────────┘ └──────────┘  └──────────┘
```

**Orchestrated by Prefect** with task-level retries, logging, and monitoring.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python run_pipeline.py
```

---

## 📁 Project Structure

```
recomart-pipeline/
├── config/pipeline_config.yaml     # Central configuration
├── src/
│   ├── ingestion/                  # Kaggle + API data ingestion
│   ├── validation/                 # Data quality checks + HTML report
│   ├── preparation/                # Cleaning, encoding, EDA plots
│   ├── transformation/             # Feature engineering
│   ├── feature_store/              # Feast integration
│   ├── versioning/                 # SHA256 hashing + lineage
│   ├── training/                   # SVD + GradientBoosting + MLflow
│   └── utils/                      # Centralized logger
├── orchestration/pipeline_flow.py  # Prefect DAG
├── run_pipeline.py                 # Main entry point
├── data/                           # Generated data (auto-created)
├── reports/                        # Quality reports + EDA plots
├── models/                         # Trained models
└── mlruns/                         # MLflow tracking
```

---

## 🔧 Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Data Storage | Local Filesystem + SQLite |
| Feature Store | Feast |
| Model Training | Scikit-learn, Surprise (SVD) |
| Experiment Tracking | MLflow |
| Orchestration | Prefect |
| Visualization | Matplotlib, Seaborn |

---

## 📊 Pipeline Stages

### Stage 1: Data Ingestion
- **Source 1**: Kaggle dataset (13 product features) via `kagglehub`
- **Source 2**: Synthetic user interactions (200 users, 5000 events)
- Retry logic, error handling, timestamped storage

### Stage 2: Data Validation
- Schema validation, missing values, duplicates, range checks
- HTML quality report with pass/fail metrics and overall score

### Stage 3: Data Preparation & EDA
- Missing value imputation, categorical encoding, numerical normalization
- 6 EDA visualizations (distributions, correlations, heatmaps)

### Stage 4: Feature Engineering
- **Product features**: click-to-purchase ratio, price bracket, brand popularity, quality score
- **User features**: activity frequency, avg rating, rating variance, purchase ratio
- Stored in CSV, Parquet, and SQLite warehouse

### Stage 5: Feature Store (Feast)
- Registered entities (product, user) and feature views
- Online + offline stores for training and inference

### Stage 6: Data Versioning
- SHA256 hash-based file versioning
- JSON lineage tracking (source → transformation → output)

### Stage 7: Model Training
- **Content-Based**: GradientBoosting on product features
- **Collaborative Filtering**: SVD on user-product ratings
- Metrics: RMSE, MAE, R², Precision@K, Recall@K, NDCG@K
- All tracked in MLflow

---

## 📈 Evaluation Metrics

| Model | Metric | Description |
|---|---|---|
| Content-Based | RMSE | Root Mean Squared Error |
| Content-Based | MAE | Mean Absolute Error |
| Content-Based | R² | Coefficient of Determination |
| Collaborative | RMSE | Cross-validated RMSE |
| Collaborative | Precision@K | Relevant items in top-K |
| Collaborative | Recall@K | Coverage of relevant items |
| Collaborative | NDCG@K | Normalized Discounted Cumulative Gain |

---

## 📝 Dataset

**Primary**: [Ecommerce Product Recommendation Dataset](https://www.kaggle.com/datasets/kartikeybartwal/ecomerce-product-recommendation-dataset) (Kaggle)

**Supplementary**: Synthetically generated user interaction data

---

## 📄 License

Apache 2.0 (as per Kaggle dataset license)
