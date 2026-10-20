"""Simple inventory policy simulation based on forecast demand.

This is a transparent teaching baseline, not a production replenishment system.
"""
import numpy as np
import pandas as pd


def simulate_inventory(df, demand_col="sales", forecast_col="prediction",
                       lead_time_days=7, review_period_days=7,
                       service_z=1.65, initial_inventory=0):
    """Estimate reorder point and illustrative stockout/holding outcomes per series.

    Requires a single time series sorted by date. For grouped simulations, call
    this function separately for each item/store series.
    """
    df = df.copy().sort_values("date" if "date" in df.columns else "d")
    demand = pd.to_numeric(df[demand_col], errors="coerce").fillna(0).to_numpy()
    forecast = pd.to_numeric(df[forecast_col], errors="coerce").fillna(0).clip(lower=0).to_numpy()

    historical_std = float(np.std(demand, ddof=1)) if len(demand) > 1 else 0.0
    mean_forecast = float(np.mean(forecast)) if len(forecast) else 0.0
    safety_stock = service_z * historical_std * np.sqrt(max(lead_time_days, 1))
    reorder_point = mean_forecast * lead_time_days + safety_stock

    stock = float(initial_inventory)
    ending_inventory, stockout_units, orders = [], [], []
    for actual in demand:
        order_qty = max(0.0, reorder_point - stock) if stock <= reorder_point else 0.0
        stock += order_qty
        fulfilled = min(stock, actual)
        shortage = max(0.0, actual - fulfilled)
        stock -= fulfilled
        ending_inventory.append(stock)
        stockout_units.append(shortage)
        orders.append(order_qty)

    result = df.copy()
    result["simulated_order_qty"] = orders
    result["ending_inventory"] = ending_inventory
    result["stockout_units"] = stockout_units
    result.attrs["summary"] = {
        "lead_time_days": lead_time_days,
        "review_period_days": review_period_days,
        "safety_stock_estimate": float(safety_stock),
        "reorder_point_estimate": float(reorder_point),
        "total_stockout_units": float(np.sum(stockout_units)),
        "total_order_qty": float(np.sum(orders)),
        "average_ending_inventory": float(np.mean(ending_inventory)) if ending_inventory else 0.0,
    }
    return result
