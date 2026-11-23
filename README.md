# Retail Demand Forecasting & Hierarchical Supply Chain Analytics

**End-to-end retail demand forecasting using Python, LightGBM, and hierarchical forecasting techniques.**

This project explores how transaction-scale retail data can be transformed into reliable demand forecasts for different planning decisions, from individual store–item combinations to total chain demand. It combines large-scale data preparation, exploratory demand analysis, statistical and machine-learning baselines, rolling backtesting, and a comparison of **Bottom-up** versus **Top-down** hierarchical forecasting.

> **Headline result:** On the 16-day final evaluation period, **Tweedie LightGBM with Bottom-up aggregation** achieved **9.55% WAPE and +0.19% signed bias at the Total level**. It also outperformed the evaluated proportional Top-down approach on RMSSE at all five hierarchy levels. These are holdout results for the **10-store development subset**, not a claim of production accuracy across the full retail chain.

## Project at a glance

| Area | Implementation |
|---|---|
| Dataset | Corporación Favorita Grocery Sales Forecasting (Kaggle) |
| Raw scale | More than 125 million recorded sales rows |
| Data preparation | Chunked ingestion, cleaning, store-day calendar, joins, compact types, Parquet |
| Development scope | 10 stores; approximately 5.9 million rows in the modelling subset |
| Forecast horizon | Rolling **16-day-ahead** demand forecasts |
| Forecasting methods | Moving-average and seasonal baselines, Global LightGBM (Log/Tweedie), proportional Top-down |
| Hierarchy | Store–Item, Family, Store, State, Total |
| Evaluation | RMSSE, WAPE, signed forecast bias; eight rolling backtesting blocks and a later final-period check |
| Main tools | Python, Pandas, NumPy, LightGBM, PyArrow, Matplotlib, Jupyter Notebook |

## 1. Business problem

Retail demand varies across products, locations, calendar periods, and buying patterns. Accurate forecasts at the *total* level can support capacity and purchasing plans, while forecasts at the *store–item* level are needed for detailed replenishment. However, a model that predicts aggregate demand well may still allocate that demand poorly among individual items.

This project investigates three questions:

1. Can machine-learning models improve on simple historical forecasting baselines?
2. How do accuracy and systematic over-/underforecasting vary across demand patterns and aggregation levels?
3. Does **Bottom-up aggregation** or **Top-down proportional allocation** produce better coherent forecasts for this retail subset?

## 2. Dataset and data pipeline

**Source:** [Corporación Favorita Grocery Sales Forecasting — Kaggle](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)

The project uses historical sales together with store and item metadata, promotions, transactions, holidays/events, and oil prices. The original training table contains more than **125 million sales records**. Data preparation constructs a larger, approximately **194 million-row** daily sales grid before a smaller development subset is selected for detailed modelling.

The processing stages are:

```text
Raw CSV files (125M+ sales rows)
        |
        v
Chunked reading, dtype optimization, Parquet staging
        |
        v
Cleaning + store-day calendar + closure/missingness checks
        |
        v
Store/item/holiday/oil enrichment and demand features
        |
        v
10-store development subset + chronological evaluation splits
        |
        v
EDA, demand segmentation, baselines, Global LightGBM
        |
        v
Bottom-up / Top-down hierarchy + backtesting + evaluation
```

**Important data treatment:** Missing transaction records and suspected store-closure days are not assumed to be ordinary zero demand without qualification. The modelling workflow also checks time-dependent features to reduce look-ahead leakage.

### Exploratory analysis

The EDA explores weekly and monthly demand trends, payday and event effects, product-family differences, sales concentration, and demand intermittency. Series are also analysed using **ADI/CV² demand patterns** and **ABC–XYZ segmentation** to show why one forecasting approach may not work equally well for every item.

## 3. Forecasting methodology

### Statistical and machine-learning benchmarks

- **Statistical baselines:** historical moving averages and naive/seasonal-naive alternatives; additional intermittent-demand benchmark experiments are included in the notebooks.
- **Global LightGBM:** models trained across many Store–Item observations, using historical lags, rolling statistics, calendar variables, and other validated features.
- **Alternative objectives:** Log-transformed regression and Tweedie regression, compared on aligned evaluation populations.

The protocol evaluates each target date using demand information available **at least 16 days earlier** for demand-derived features. It is a **rolling 16-day-ahead** setup; it should not be confused with one forecast issued at a single origin for all 16 future dates.

### Hierarchical forecasting

The project evaluates five grouping levels:

```text
                      Total
                    /   |   \
               Store   State   Family    <- Alternative aggregations
                    \   |   /
                    Store–Item            <- Lowest-level forecast units
```

*Store, State, and Family are overlapping aggregation views of Store–Item sales, rather than a single sequential path.*

**Bottom-up (BU):** A Global LightGBM model predicts each Store–Item series, then the predictions are summed to obtain Family, Store, State, and Total forecasts.

**Top-down (TD):** An independent LightGBM model first predicts **Total** demand. That forecast is distributed across Store–Item series using non-negative historical sales proportions (the primary specification uses a **90-day** allocation window). Allocated predictions are subsequently summed to the other levels.

Both methods produce **coherent** forecasts: the individual Store–Item predictions reconcile with the corresponding aggregate forecasts. Coherence, however, does **not** guarantee forecast accuracy.

## 4. Evaluation design

- **Rolling backtesting:** eight historical blocks, each with a 16-day prediction window, to compare performance across different periods.
- **Allocation sensitivity:** 30-, 90-, 180-, and 365-day historical-share windows for Top-down.
- **Training sensitivity:** 90-, 180-, and 365-day training windows for the independent Total model; the selected primary specification uses 180 days.
- **Final-period evaluation:** **31 July–15 August 2017**, covering **4,378 eligible Store–Item series** and **70,048 Store–Item/date predictions**.
- **Consistency audit:** Final-period Store–Item metrics from the hierarchy notebook were checked against earlier model-evaluation results.

### Error metrics

| Metric | Interpretation |
|---|---|
| **RMSSE** (Root Mean Squared Scaled Error) | Squared forecast error scaled by historical one-day demand changes; **lower is better**. Results at grouped levels report the mean of eligible group-level RMSSE values. |
| **WAPE** (Weighted Absolute Percentage Error) | `sum(abs(forecast - actual)) / sum(actual) × 100`; **lower is better**. |
| **Signed bias** | `sum(forecast - actual) / sum(actual) × 100`; positive means net overforecast, negative means net underforecast. Values near zero can still mask offsetting item-level errors. |

## 5. Results

### Final-period model comparison — Store–Item

| Model | RMSSE ↓ | WAPE ↓ | Signed bias |
|---|---:|---:|---:|
| Moving Average 28 | 0.7273 | 58.77% | +6.90% |
| **Log LightGBM Bottom-up** | **0.6973** | **50.82%** | −21.21% |
| Tweedie LightGBM Bottom-up | 0.7081 | 52.51% | **+0.19%** |
| Tweedie Top-down | 0.7484 | 63.47% | +11.09% |

**Interpretation:** Log LightGBM achieved the lowest Store–Item RMSSE, but underforecast total demand substantially. Tweedie Bottom-up provided a more balanced trade-off between item-level accuracy and aggregate signed bias.

### Final-period hierarchical RMSSE

| Hierarchy level | Tweedie Bottom-up ↓ | Tweedie Top-down ↓ |
|---|---:|---:|
| Store–Item | **0.7081** | 0.7484 |
| Family | **0.8900** | 0.9634 |
| Store | **0.6353** | 0.9502 |
| State | **0.7646** | 1.1585 |
| Total | **0.7362** | 1.0315 |

At the **Total** level, Bottom-up Tweedie recorded **9.55% WAPE / +0.19% signed bias**, compared with **14.48% WAPE / +11.09% signed bias** for proportional Top-down.

### Eight-block rolling backtesting — mean RMSSE

| Hierarchy level | Tweedie Bottom-up ↓ | Tweedie Top-down ↓ |
|---|---:|---:|
| Store–Item | **0.7550** | 0.7851 |
| Family | **0.7765** | 0.8923 |
| Store | **0.9047** | 1.0640 |
| State | **0.9353** | 1.0575 |
| Total | **0.7569** | 0.8702 |

The eight-block comparison and later evaluation period support the same overall conclusion **for the tested configurations**: Bottom-up Tweedie performs better on hierarchical RMSSE. This is not evidence that Bottom-up universally outperforms every possible Top-down or forecast-reconciliation method.

### Saved visualizations

The hierarchical forecasting notebook writes these figures when run:

- `figures/hierarchical/bottomup_vs_topdown_rmsse.png`
- `figures/hierarchical/allocation_window_sensitivity.png`
- `figures/hierarchical/training_window_sensitivity.png`

## 6. Notebook guide

Run the notebooks in numerical order. The filenames below correspond to the project notebooks; ensure the files are present in the same workspace before executing the full pipeline.

| Notebook | Main responsibility |
|---|---|
| `01_load_and_profile.ipynb` | Profile the raw data and process large sales files in chunks |
| `02_cleaning_and_join.ipynb` | Build the daily sales grid, dimension tables, and data-quality checks |
| `03_features.ipynb` | Create modelling subset and leakage-aware forecasting features |
| `04_eda.ipynb` | Explore seasonality, demand patterns, and ABC–XYZ segments |
| `05_baseline.ipynb` | Calculate statistical forecasts and baseline metrics |
| `06_lightgbm_model.ipynb` | Train/evaluate Global LightGBM and run rolling backtests |
| `07_hierarchical_forecasting.ipynb` | Implement Bottom-up/Top-down forecasts, sensitivity tests, and final hierarchical evaluation |

The repository may also contain cleaned notebook revisions or audit copies; use the latest validated version of each notebook when reproducing the results.

## 7. Project files

The notebooks use this **local workspace layout** (large generated datasets are not necessarily included in a public GitHub repository):

```text
~/Projects/bigdata/
├── raw/                         # Original Kaggle CSV files
├── interim/train/               # Chunked Parquet staging
├── clean/dim/                   # Cleaned dimensions and store-day calendar
├── features/                    # Modelling features and specifications
├── results/
│   ├── rolling_backtest/        # Block-level forecasts and evaluation
│   ├── hierarchical_forecasting/# Hierarchical results and holdout outputs
│   └── global_lgbm_25_predictions.parquet
└── figures/
    └── hierarchical/            # Comparison and sensitivity charts
```

Notable outputs include:

- `results/hierarchical_forecasting/final_holdout_hierarchy_metrics.csv` — RMSSE, WAPE, and signed bias across all five levels.
- `results/hierarchical_forecasting/final_holdout_store_item_predictions.parquet` — aligned Store–Item forecasts for the final-period comparison.
- `results/hierarchical_forecasting/final_holdout_hierarchy_predictions.parquet` — aggregated forecasts and historical RMSSE scales.
- `results/hierarchical_forecasting/final_holdout_total_forecasts.csv` — independent Total-level predictions.

## 8. Running the project

**Prerequisites:** Python, Jupyter Notebook, and the packages used in the notebooks. Core dependencies include `pandas`, `numpy`, `pyarrow`, `lightgbm`, `scikit-learn`, and `matplotlib`. Some experiment cells also use `joblib`.

```bash
python -m pip install jupyter pandas numpy pyarrow lightgbm scikit-learn matplotlib joblib
jupyter notebook
```

1. Download the dataset from Kaggle and extract the expected CSV files into `~/Projects/bigdata/raw/` (including `train.csv` and the companion metadata tables).
2. Update `BASE = Path.home() / "Projects/bigdata"` in the notebooks if using another directory.
3. Run notebooks **01 → 07** in order, retaining generated Parquet datasets and intermediate results between notebooks.
4. Review the saved evaluation CSVs and charts in `results/` and `figures/`.

**Reproducibility note:** This is a notebook-based research workflow, not a one-command production pipeline. Some downstream notebooks depend on artifacts written by earlier experiments, and the full-data preprocessing stage requires sufficient memory and disk space. Dataset files and generated results should be handled separately from source code when publishing on GitHub.

## 9. Limitations and practical relevance

- **Development scope:** Detailed machine-learning and hierarchical experiments use a 10-store subset, not all 54 stores. Some baseline experiments also examine wider coverage.
- **Evaluation cohort:** The final 4,378 series were selected using complete evaluation coverage and valid historical RMSSE scales; this is a restricted evaluation population.
- **Holdout interpretation:** The final period is a useful later-period confirmation of the specified hierarchical approach, but earlier model-development work also examined this timeframe. It should not be described as completely untouched for the overall project.
- **Top-down design:** Only historical-proportion allocation from an independently forecast Total is considered here; methods such as MinT reconciliation were not evaluated.
- **Operational deployment:** The project does not test inventory costs, stockouts, replenishment lead times, or an automated ordering system. Strong aggregate accuracy does not automatically imply accurate SKU-level replenishment quantities.
- **Temporal generalization:** Historical results from 2017 do not establish performance on current retail demand without retraining and monitoring.

**Practical takeaway:** Bottom-up Tweedie is a promising candidate for **forecasting-assisted planning at aggregated levels** in this evaluated subset. Further validation is required before using the forecasts for automated store-level replenishment.

## 10. Next steps

Potential extensions include wider-store model validation, segment-specific forecasting policies, operational inventory simulations using explicitly stated assumptions, automated model monitoring, and evaluation of alternative hierarchical reconciliation methods.

---

*Academic portfolio project — research and decision-support prototype, not a production demand-planning system.*
