# Retail Demand Forecasting & Inventory Simulation (M5)

A learning project built around the public M5 Walmart retail sales dataset. The repository is organized as a small, modular forecasting pipeline:

1. **Data engineering** — load M5 sales, calendar, and optional weekly price data; reshape daily sales from wide to long format.
2. **Feature engineering** — calendar variables, lag features, and rolling averages.
3. **Model training** — time-based train/validation split and a scikit-learn gradient-boosting baseline.
4. **Evaluation** — MAE, RMSE, and WAPE.
5. **Supply-chain simulation** — an illustrative reorder-point policy with safety-stock and stockout outputs.
6. **Dashboard** — optional Streamlit app to inspect exported forecast and inventory outputs.

## Dataset

Use the official M5 Forecasting dataset from Kaggle:
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Download and place these files in `data/raw/`:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv` (optional for the current baseline)

Do not commit the raw dataset to GitHub unless you have checked the dataset terms and repository size. The `.gitignore` excludes raw data by default.

Expected project layout:

```text
m5_retail_demand_forecasting/
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── config.py
├── module1_data_engineering.py
├── module2_feature_engineering.py
├── module3_model_training.py
├── module4_evaluation.py
├── module5_supply_chain.py
├── requirements.txt
└── run_pipeline.py
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\\Scripts\\activate

pip install -r requirements.txt
```

Place the dataset files in `data/raw/`. For a quick smoke test, set `MAX_TRAIN_ROWS = 10` or `50` in `config.py`. The M5 dataset is large, and reshaping all item-store series into long format can require substantial memory. Start small before scaling up.

Run:

```bash
python run_pipeline.py
```

Outputs are written to `outputs/`:
- `forecast.csv`
- `evaluation_metrics.json`
- `inventory_simulation.csv`

## Important limitations

- This is a baseline implementation and has not been benchmarked against the official M5 leaderboard.
- The validation split is time-based, but the current script evaluates rows from the last validation window rather than producing a recursive 28-day forecast for every series.
- The inventory simulation is illustrative. It does not yet model delayed replenishment arrivals, per-SKU lead times, holding costs, lost-sales/backorder differences, or a full service-level optimization.
- The current pipeline is designed for small experiments first. Full M5-scale runs may require memory optimization, chunking, feature-store style processing, or a database-backed workflow.

## Next improvements

- Add a seasonal-naive baseline and compare it against gradient boosting.
- Implement rolling-origin backtesting.
- Add WRMSSE to align evaluation more closely with the M5 competition.
- Train at item-store level and reconcile forecasts across item, department, category, store, and state levels.
- Improve inventory simulation with lead-time demand, replenishment delays, holding costs, and service-level targets.
- Record reproducible experiments and document your own contributions.

## Attribution and responsible use

If this repository is adapted from an existing project, keep the original author's attribution and comply with the applicable license. Clearly document which parts were reused and which parts you personally changed, tested, and validated. Do not claim benchmark results unless you reproduced them.
