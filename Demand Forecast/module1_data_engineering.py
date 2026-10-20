"""Load and reshape M5 data into a daily item-store time series."""
from pathlib import Path
import pandas as pd


def load_m5_data(sales_path, calendar_path, prices_path=None):
    sales_path = Path(sales_path)
    calendar_path = Path(calendar_path)
    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path) if prices_path and Path(prices_path).exists() else None
    return sales, calendar, prices


def reshape_sales(sales, calendar, max_rows=None):
    """Convert M5's wide daily sales columns to long format.

    max_rows is intended for quick local experiments; use None for all series.
    """
    sales = sales.copy()
    id_cols = [
        col for col in
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        if col in sales.columns
    ]
    day_cols = [col for col in sales.columns if col.startswith("d_")]
    if not day_cols:
        raise ValueError("No daily sales columns like d_1, d_2, ... were found.")

    if max_rows is not None:
        sales = sales.head(max_rows)

    long_df = sales.melt(
        id_vars=id_cols, value_vars=day_cols,
        var_name="d", value_name="sales"
    )
    long_df["sales"] = pd.to_numeric(long_df["sales"], errors="coerce").fillna(0)
    long_df = long_df.merge(calendar, on="d", how="left", validate="many_to_one")
    if "date" in long_df.columns:
        long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    return long_df


def merge_prices(long_df, prices):
    """Optionally join weekly item-store selling prices."""
    if prices is None:
        return long_df
    if "wm_yr_wk" not in long_df.columns:
        raise ValueError("Calendar data must contain wm_yr_wk to join sell_prices.")
    keys = [k for k in ["store_id", "item_id", "wm_yr_wk"] if k in long_df.columns and k in prices.columns]
    if len(keys) != 3:
        raise ValueError("Expected store_id, item_id, and wm_yr_wk in both datasets.")
    return long_df.merge(prices, on=keys, how="left", validate="many_to_one")


def save_processed(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
