"""Run a lightweight M5 pipeline.

Expected files under data/raw:
  sales_train_validation.csv
  calendar.csv
  sell_prices.csv (optional)

For a quick test, set MAX_TRAIN_ROWS in config.py to a small number.
"""
import json
import pandas as pd

import config
from module1_data_engineering import load_m5_data, reshape_sales, merge_prices
from module2_feature_engineering import add_time_features, add_lag_rolling_features
from module3_model_training import train_and_predict
from module4_evaluation import evaluate_forecasts
from module5_supply_chain import simulate_inventory


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sales, calendar, prices = load_m5_data(
        config.SALES_FILE, config.CALENDAR_FILE, config.PRICES_FILE
    )
    long_df = reshape_sales(sales, calendar, max_rows=config.MAX_TRAIN_ROWS)
    long_df = merge_prices(long_df, prices)
    long_df = add_time_features(long_df)

    # M5 has a very long daily history. Feature generation is memory intensive;
    # run on a limited number of series first, then scale up.
    feature_df = add_lag_rolling_features(long_df)
    feature_df = feature_df.dropna(subset=["lag_7", "rolling_mean_7"]).copy()

    model, validation, metrics, feature_cols = train_and_predict(
        feature_df, validation_days=config.VALIDATION_DAYS
    )
    validation["prediction"] = validation["prediction"].clip(lower=0)
    metrics.update(evaluate_forecasts(validation["sales"], validation["prediction"]))

    validation.to_csv(config.FORECAST_OUTPUT, index=False)
    with open(config.METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Run an illustrative simulation for the first series in validation.
    group_col = "id" if "id" in validation.columns else None
    sample = validation
    if group_col:
        sample = validation.loc[validation[group_col] == validation[group_col].iloc[0]]
    simulation = simulate_inventory(sample, demand_col="sales", forecast_col="prediction")
    simulation.to_csv(config.INVENTORY_OUTPUT, index=False)

    print("Pipeline complete.")
    print(f"Forecast output: {config.FORECAST_OUTPUT}")
    print(f"Metrics output: {config.METRICS_OUTPUT}")
    print(f"Inventory output: {config.INVENTORY_OUTPUT}")
    print(json.dumps(metrics, indent=2))
    print(f"Features used: {feature_cols}")


if __name__ == "__main__":
    main()
