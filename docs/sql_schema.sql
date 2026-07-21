-- ============================================================
-- RecoMart Pipeline - Data Warehouse SQL Schema
-- Database: SQLite (data/warehouse.db)
-- ============================================================


-- ------------------------------------------------------------
-- Table: product_features
-- Source: data/features/product_features.csv
-- Description: Engineered product-level features derived from
--              the Kaggle Ecommerce Recommendation Dataset.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS product_features (
    product_id                                              INTEGER PRIMARY KEY,

    -- Raw Kaggle columns
    "Number of clicks on similar products"                  REAL,
    "Number of similar products purchased so far"           REAL,
    "Average rating given to similar products"              REAL,
    "Gender"                                                INTEGER,    -- Label-encoded
    "Median purchasing price (in rupees)"                   REAL,
    "Rating of the product"                                 REAL,       -- 0.0 - 5.0
    "Brand of the product"                                  INTEGER,    -- Label-encoded
    "Customer review sentiment score (overall)"             REAL,       -- -1.0 to 1.0
    "Price of the product"                                  REAL,
    "Holiday"                                               INTEGER,    -- 0 = No, 1 = Yes
    "Season"                                                INTEGER,    -- Label-encoded
    "Geographical locations"                                INTEGER,    -- Label-encoded
    "Probability for the product to be recommended"         REAL,       -- TARGET column (0.0 - 1.0)

    -- Engineered features
    click_to_purchase_ratio                                 REAL,       -- clicks / (purchases + 1)
    price_bracket                                           INTEGER,    -- Quartile: 0=Low, 1=Med, 2=High, 3=Premium
    brand_popularity                                        REAL,       -- Normalised brand demand score
    quality_score                                           REAL,       -- 0.7 * rating + 0.3 * sentiment
    seasonal_demand                                         TEXT,       -- "{season}_{holiday}" e.g. "2_1"

    -- Short aliases (for model training convenience)
    rating                                                  REAL,       -- Alias for "Rating of the product"
    sentiment_score                                         REAL,       -- Alias for "Customer review sentiment score"
    price                                                   REAL,       -- Alias for "Price of the product"

    -- Feast compatibility
    event_timestamp                                         TIMESTAMP
);


-- ------------------------------------------------------------
-- Table: user_features
-- Source: data/features/user_features.csv
-- Description: Engineered user-level behavioural features
--              derived from synthetic interaction data.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_features (
    user_id                     INTEGER PRIMARY KEY,

    -- Engineered features
    activity_frequency          INTEGER,    -- Total number of interactions by user
    avg_rating_given            REAL,       -- Mean rating across rated interactions
    rating_variance             REAL,       -- Std dev of ratings (preference consistency)
    unique_items_interacted     INTEGER,    -- Count of distinct products interacted with
    purchase_ratio              REAL,       -- purchases / total_interactions

    -- Feast compatibility
    event_timestamp             TIMESTAMP
);


-- ------------------------------------------------------------
-- Table: interaction_matrix
-- Source: data/features/interaction_matrix.csv
-- Description: Long-format user x item interaction matrix
--              used as input to the SVD collaborative filtering model.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS interaction_matrix (
    user_id     INTEGER NOT NULL,
    product_id  INTEGER NOT NULL,
    rating      REAL,               -- Explicit rating 1.0 - 5.0 (NULL if not rated)
    PRIMARY KEY (user_id, product_id),
    FOREIGN KEY (product_id) REFERENCES product_features(product_id),
    FOREIGN KEY (user_id)    REFERENCES user_features(user_id)
);


-- ============================================================
-- Indexes for query performance
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_interaction_user    ON interaction_matrix(user_id);
CREATE INDEX IF NOT EXISTS idx_interaction_product ON interaction_matrix(product_id);
CREATE INDEX IF NOT EXISTS idx_product_rating      ON product_features(rating);
CREATE INDEX IF NOT EXISTS idx_product_brand       ON product_features("Brand of the product");
