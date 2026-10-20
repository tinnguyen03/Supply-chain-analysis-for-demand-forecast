"""Evaluation helpers for point forecasts."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecasts(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    denominator = np.abs(actual).sum()
    wape = float(np.abs(actual - predicted).sum() / denominator) if denominator else None
    return {"MAE": float(mae), "RMSE": rmse, "WAPE": wape}


def evaluate_by_group(df, group_col="cat_id", actual_col="sales", pred_col="prediction"):
    if group_col not in df.columns:
        return None
    rows = []
    for group, part in df.groupby(group_col):
        metrics = evaluate_forecasts(part[actual_col], part[pred_col])
        rows.append({group_col: group, **metrics})
    return rows
