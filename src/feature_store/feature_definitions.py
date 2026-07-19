import os
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

# Dynamically resolve project root from this file's location (portable across machines)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PRODUCT_FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features", "product_features.parquet")
USER_FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features", "user_features.parquet")

product_entity = Entity(name="product_id", value_type=ValueType.INT64)
user_entity = Entity(name="user_id", value_type=ValueType.INT64)

product_features_source = FileSource(
    path=PRODUCT_FEATURES_PATH,
    timestamp_field="event_timestamp"
)

user_features_source = FileSource(
    path=USER_FEATURES_PATH,
    timestamp_field="event_timestamp"
)

product_features_view = FeatureView(
    name="product_features",
    entities=[product_entity],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="click_to_purchase_ratio", dtype=Float64),
        Field(name="price_bracket", dtype=Int64),
        Field(name="brand_popularity", dtype=Float64),
        Field(name="quality_score", dtype=Float64),
        Field(name="rating", dtype=Float64),
        Field(name="sentiment_score", dtype=Float64),
        Field(name="price", dtype=Float64),
    ],
    online=True,
    source=product_features_source,
    tags={},
)

user_features_view = FeatureView(
    name="user_features",
    entities=[user_entity],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="activity_frequency", dtype=Int64),
        Field(name="avg_rating_given", dtype=Float64),
        Field(name="rating_variance", dtype=Float64),
        Field(name="unique_items_interacted", dtype=Int64),
        Field(name="purchase_ratio", dtype=Float64),
    ],
    online=True,
    source=user_features_source,
    tags={},
)
