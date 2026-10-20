"""Train a baseline and optional LightGBM model using time-based validation."""
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor


def time_split(df, validation_days=28, date_col="date"):
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Missing required column: {date_col}")
    dates = pd.to_datetime(df[date_col])
    cutoff = dates.max() - pd.Timedelta(days=validation_days - 1)
    return df.loc[dates < cutoff].copy(), df.loc[dates >= cutoff].copy()


def select_feature_columns(df):
    excluded = {
        "sales", "date", "d", "id", "item_id", "dept_id", "cat_id",
        "store_id", "state_id", "event_name_1", "event_name_2",
        "event_type_1", "event_type_2"
    }
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def train_model(train_df, feature_cols):
    model = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
        random_state=42
    )
    model.fit(train_df[feature_cols].fillna(0), train_df["sales"])
    return model


def train_and_predict(feature_df, validation_days=28):
    train_df, valid_df = time_split(feature_df, validation_days=validation_days)
    feature_cols = select_feature_columns(feature_df)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for training.")
    model = train_model(train_df, feature_cols)
    predictions = model.predict(valid_df[feature_cols].fillna(0))
    predictions = predictions.clip(min=0)
    mae = mean_absolute_error(valid_df["sales"], predictions)
    result = valid_df.copy()
    result["prediction"] = predictions
    return model, result, {"MAE": float(mae), "validation_rows": int(len(valid_df))}, feature_cols
