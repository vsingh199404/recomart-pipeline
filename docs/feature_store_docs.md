# Task 7: Feature Store Documentation (Feast)

## Overview
The RecoMart pipeline uses **Feast** (Feature Store) to manage feature versioning, offline training data retrieval, and low-latency online feature serving for the recommendation models.

## Configuration (`src/feature_store/feature_store.yaml`)

```yaml
project: recomart
provider: local
offline_store:
  type: file           # Parquet files as offline source
online_store:
  type: sqlite         # SQLite for low-latency online serving
  path: data/feast/online_store.db
```

---

## Entities

| Entity Name | Type | Description |
|---|---|---|
| `product_id` | INT64 | Unique product identifier (from Kaggle dataset) |
| `user_id` | INT64 | Unique user identifier (from synthetic interactions) |

---

## Feature Views

### `product_features`
**Source:** `data/features/product_features.parquet`
**Entity:** `product_id`
**TTL:** 3650 days

| Feature Name | Type | Description | Source Column |
|---|---|---|---|
| `click_to_purchase_ratio` | Float64 | Conversion signal | Engineered |
| `price_bracket` | Int64 | Price quartile (0–3) | Engineered |
| `brand_popularity` | Float64 | Normalized brand demand | Engineered |
| `quality_score` | Float64 | 0.7×rating + 0.3×sentiment | Engineered |
| `Rating of the product` | Float64 | Direct product rating | Kaggle |
| `Customer review sentiment score (overall)` | Float64 | Sentiment score | Kaggle |
| `Price of the product` | Float64 | Product price (INR) | Kaggle |
| `Probability for the product to be recommended to the person` | Float64 | Recommendation target | Kaggle |

---

### `user_features`
**Source:** `data/features/user_features.parquet`
**Entity:** `user_id`
**TTL:** 3650 days

| Feature Name | Type | Description | Source Column |
|---|---|---|---|
| `activity_frequency` | Int64 | Total interactions | Engineered |
| `avg_rating_given` | Float64 | Mean rating given | Engineered |
| `rating_variance` | Float64 | Std dev of ratings | Engineered |
| `unique_items_interacted` | Int64 | Distinct products seen | Engineered |
| `purchase_ratio` | Float64 | Purchase intent ratio | Engineered |

---

## Feature Store Operations

| Operation | Method | Description |
|---|---|---|
| **Apply** | `store.apply([entities, feature_views])` | Register entities and views |
| **Materialize** | `store.materialize_incremental(end_date)` | Push offline → online store |
| **Get Historical** | `store.get_historical_features(entity_df, features)` | Training data retrieval |
| **Get Online** | `store.get_online_features(features, entity_rows)` | Real-time inference |

---

## File Locations

| File | Purpose |
|---|---|
| `src/feature_store/feature_definitions.py` | Entity and FeatureView definitions |
| `src/feature_store/store.py` | All Feast operations (setup, materialize, serve) |
| `src/feature_store/feature_store.yaml` | Feast project configuration |
| `data/features/product_features.parquet` | Offline source for product features |
| `data/features/user_features.parquet` | Offline source for user features |
| `data/feast/online_store.db` | SQLite online store (materialized) |

---

## How to Use

```python
from src.feature_store.store import get_online_features

# Real-time feature retrieval for inference
entity_rows = [{"product_id": 42}, {"product_id": 101}]
features = get_online_features(config, entity_rows)
```
