# Task 6: SQL Schema & Feature Logic Summary

## Data Warehouse — SQLite Schema (`data/warehouse.db`)

### Table: `product_features`
Stores engineered product-level features derived from the Kaggle dataset.

```sql
CREATE TABLE IF NOT EXISTS product_features (
    product_id                  INTEGER PRIMARY KEY,
    -- Raw Kaggle columns
    clicks                      REAL,     -- Number of clicks on similar products
    purchases                   REAL,     -- Number of similar products purchased
    avg_rating                  REAL,     -- Average rating given to similar products
    gender                      INTEGER,  -- Label-encoded gender
    median_price                REAL,     -- Median purchasing price (INR)
    rating                      REAL,     -- Rating of the product
    brand                       INTEGER,  -- Label-encoded brand
    sentiment_score             REAL,     -- Customer review sentiment score
    price                       REAL,     -- Price of the product
    holiday                     INTEGER,  -- Holiday flag (0/1)
    season                      INTEGER,  -- Label-encoded season
    geo_location                INTEGER,  -- Label-encoded geographical location
    recommendation_probability  REAL,     -- Target: probability of recommendation
    -- Engineered features
    click_to_purchase_ratio     REAL,     -- clicks / (purchases + 1)
    price_bracket               INTEGER,  -- Quartile-based: 0=Low, 1=Med, 2=High, 3=Premium
    brand_popularity            REAL,     -- Normalized count of products per brand
    quality_score               REAL,     -- 0.7 * rating + 0.3 * sentiment_score
    seasonal_demand             TEXT,     -- season + '_' + holiday (e.g. '2_1')
    event_timestamp             TIMESTAMP -- Feast compatibility timestamp
);
```

### Table: `user_features`
Stores engineered user-level behavioral features from synthetic interaction data.

```sql
CREATE TABLE IF NOT EXISTS user_features (
    user_id                  INTEGER PRIMARY KEY,
    activity_frequency       INTEGER,  -- Total interactions by user
    avg_rating_given         REAL,     -- Mean rating across rated interactions
    rating_variance          REAL,     -- Std dev of ratings (engagement consistency)
    unique_items_interacted  INTEGER,  -- Count of distinct products interacted with
    purchase_ratio           REAL,     -- purchases / total_interactions
    event_timestamp          TIMESTAMP -- Feast compatibility timestamp
);
```

### Table: `interaction_matrix`
Long-format user × item interaction matrix for collaborative filtering.

```sql
CREATE TABLE IF NOT EXISTS interaction_matrix (
    user_id     INTEGER,
    product_id  INTEGER,
    rating      REAL,
    PRIMARY KEY (user_id, product_id)
);
```

---

## Feature Logic Summary

### Product Features

| Feature | Formula / Logic | Purpose |
|---|---|---|
| `click_to_purchase_ratio` | `clicks / (purchases + 1)` | Conversion signal — high ratio = interest without conversion |
| `price_bracket` | Quartile of price column (0–3) | Price sensitivity segmentation |
| `brand_popularity` | `brand_count / total_products` (normalized) | Brand demand signal |
| `quality_score` | `0.7 × rating + 0.3 × sentiment_score` | Composite product quality index |
| `seasonal_demand` | `f"{season}_{holiday}"` | Interaction of seasonal and holiday context |
| `rating` | Alias for `Rating of the product` | Direct product rating |
| `sentiment_score` | Alias for `Customer review sentiment score (overall)` | Customer sentiment |
| `price` | Alias for `Price of the product` | Product price |

### User Features

| Feature | Formula / Logic | Purpose |
|---|---|---|
| `activity_frequency` | `COUNT(interactions)` per user | Overall engagement level |
| `avg_rating_given` | `MEAN(rating)` for rated interactions | User's rating tendency |
| `rating_variance` | `STD(rating)` per user | Consistency of user preferences |
| `unique_items_interacted` | `COUNT(DISTINCT product_id)` | Breadth of exploration |
| `purchase_ratio` | `purchase_count / total_interactions` | Purchase intent signal |

### Interaction Matrix

| Field | Description |
|---|---|
| `user_id` | Integer-encoded user (U001→1, U002→2, …) |
| `product_id` | Product index from Kaggle dataset |
| `rating` | Explicit rating (1.0–5.0) for rated interactions |

---

## Transformation Scripts

| Script | Responsibility |
|---|---|
| `src/transformation/feature_engineer.py` | All feature creation, Parquet/CSV/SQLite storage |
| `src/preparation/data_preparer.py` | Cleaning, Label Encoding, MinMaxScaler |
| `config/pipeline_config.yaml` | Paths for all input/output directories |
