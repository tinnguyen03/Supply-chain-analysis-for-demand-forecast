"""Project configuration for the M5 retail demand forecasting pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

SALES_FILE = RAW_DIR / "sales_train_validation.csv"
CALENDAR_FILE = RAW_DIR / "calendar.csv"
PRICES_FILE = RAW_DIR / "sell_prices.csv"

TARGET_DAYS = 28
VALIDATION_DAYS = 28
RANDOM_STATE = 42
MAX_TRAIN_ROWS = None  # Set an integer for a smaller experiment.
FORECAST_OUTPUT = OUTPUT_DIR / "forecast.csv"
METRICS_OUTPUT = OUTPUT_DIR / "evaluation_metrics.json"
INVENTORY_OUTPUT = OUTPUT_DIR / "inventory_simulation.csv"
