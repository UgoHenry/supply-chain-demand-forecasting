# Supply Chain Weekly Demand Forecasting (Moving Average Baseline vs. Gradient Boosting)

## 1. Business Objective
Forecast weekly e-commerce and retail order demand to prevent inventory stockouts during Q4 seasonal surges and reduce overstocking during off-peak weeks. This project benchmarks a traditional **4-Week Rolling Moving Average** (typical SQL/Excel planning baseline) against a supervised **Gradient Boosting Regressor** across **209 weeks** (`2015-01-04` to `2018-12-30`).

## 2. Repository Contents
* **`demand_forecasting_dual_path.ipynb`**: End-to-end Google Colab notebook covering weekly time-series aggregation, strict no-leakage lag feature engineering (`Lag_1W`, `Lag_4W`, `Rolling_4W`, `Month`, `Quarter`, `Week_of_Year`), chronological 80/20 train-test splitting, and dual-model evaluation.
* **`Weekly_Demand_Forecast.csv`**: Holdout test set comparison (`2018-03` to `2018-12`) contrasting actual weekly revenue against Path 1 (Moving Average) and Path 2 (Gradient Boosting ML) predictions.
* **`demand_forecast_chart.png`**: Time-series visualization comparing actual weekly sales against both forecasting methods across seasonal peaks and troughs.

## 3. Dual-Track Comparison: 4-Week Moving Average vs. Machine Learning (Holdout Test Set)
| Approach | Methodology | Mean Absolute Error (MAE) | Mean Absolute Percentage Error (MAPE) | Key Operational Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| **Path 1: Normal 4-Week Moving Average** | Simple rolling mean of prior 4 weeks (`Rolling_4W`) | **EUR 5,780.67** | **46.5%** | Lags behind sudden seasonal demand spikes and over-predicts immediately after Q4 holiday peaks. |
| **Path 2: Gradient Boosting Regressor** | Supervised ensemble (`200` trees, `max_depth=4`, `lr=0.05`) using calendar seasonality & lag features | **EUR 5,401.21** | **33.1%** | **+6.6% reduction in MAE** (`-13.4` percentage points in MAPE), anticipating seasonal inflection points earlier. |

## 4. Key Insights & Supply Chain Recommendations
* **Capturing Seasonality Reduces Percentage Error by 13.4 Points:** Because a 4-week moving average only looks backward, it suffers a `46.5%` MAPE during volatile retail transitions. Adding calendar features (`Month`, `Quarter`, `Week_of_Year`) alongside multi-week lags lowers MAPE to `33.1%`.
* **Inventory & Fulfillment Application:** Use the Gradient Boosting weekly forecast to trigger safety-stock replenishment 3–4 weeks ahead of Q4 peak demand windows (`October–December`), while scaling back carrier capacity commitments in January.

## 5. Tech Stack & Dataset
* **Tools:** Python (`pandas`, `numpy`, `scikit-learn`, `matplotlib`), Google Colab
* **Dataset:** [Superstore Sales Dataset (`rohitsahoo/sales-forecasting` — Kaggle)](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)
