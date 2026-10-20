"""Time-aware feature engineering for daily retail demand."""
import pandas as pd


def add_time_features(df):
    df = df.copy()
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["dayofweek"] = dt.dt.dayofweek
        df["weekofyear"] = dt.dt.isocalendar().week.astype("int16")
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    return df


def add_lag_rolling_features(df, group_cols=None, target_col="sales",
                             lags=(7, 14, 28), windows=(7, 28)):
    """Create leakage-safe lag/rolling features, assuming rows are chronological."""
    df = df.copy()
    group_cols = group_cols or [c for c in ["id"] if c in df.columns]
    sort_cols = group_cols + (["date"] if "date" in df.columns else ["d"])
    df = df.sort_values(sort_cols)
    grouped = df.groupby(group_cols, sort=False)[target_col] if group_cols else None

    if grouped is not None:
        for lag in lags:
            df[f"lag_{lag}"] = grouped.shift(lag)
        shifted = df.groupby(group_cols, sort=False)[target_col].shift(1)
        for window in windows:
            df[f"rolling_mean_{window}"] = (
                shifted.groupby([df[c] for c in group_cols], sort=False)
                .rolling(window).mean().reset_index(level=list(range(len(group_cols))), drop=True)
            )
    else:
        for lag in lags:
            df[f"lag_{lag}"] = df[target_col].shift(lag)
        shifted = df[target_col].shift(1)
        for window in windows:
            df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()

    return df


def build_features(df):
    df = add_time_features(df)
    if "sales" in df.columns:
        df = add_lag_rolling_features(df)
    return df
