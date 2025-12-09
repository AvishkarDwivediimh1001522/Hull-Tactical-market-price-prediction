# Hull Market Price Prediction & Strategy Optimization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20Pandas-orange)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## 📌 Project Overview
This project implements a machine learning pipeline to predict **Market Forward Excess Returns** using the Hull market data. Beyond simple prediction, the project includes a strategy optimization layer that dynamically adjusts leverage ($k$) to maximize a custom risk-adjusted Sharpe Ratio.

The workflow is designed to handle financial time-series data strictly, preventing look-ahead bias during validation.

## ⚙️ Technical Architecture

### 1. Data Pipeline & Cleaning
The model ingests raw `train.csv` and `test.csv` files.
* **Time-Series Ordering:** Data is explicitly sorted by `date_id` to ensure chronological integrity.
* **Imputation Strategy:**
    1.  **Forward Fill (`ffill`):** Propagates the last valid observation forward (standard for financial time series to handle gaps).
    2.  **Fill Zero:** Remaining `NaN` values at the start of the series are filled with `0`.

### 2. Feature Engineering
* **Input Features:** Includes feature groups **D, E, I, M, P, V**.
* **Exclusions:** ID columns (`date_id`), raw return targets, and non-predictive metadata (`is_scored`, `risk_free_rate`) are removed from the training feature set.
* **Target Variable:** `market_forward_excess_returns`.

### 3. Validation Strategy (Crucial)
To strictly avoid **Look-Ahead Bias**, standard K-Fold cross-validation is **not** used.
* **Split Method:** Time Series Split.
* **Ratio:** First **85%** used for Training (Historical) / Last **15%** used for Validation (Future).
* **Metadata Preservation:** `valid_meta` is extracted separately to calculate financial metrics (Sharpe Ratio) accurately during the evaluation phase.

## 🧠 Model Details
The core predictor is a **HistGradientBoostingRegressor** (Scikit-Learn). This estimator was chosen for its speed on tabular data and native handling of non-linear financial patterns.

**Hyperparameters:**
| Parameter | Value | Reasoning |
| :--- | :--- | :--- |
| `max_iter` | 300 | Sufficient boosting rounds to learn complex patterns. |
| `learning_rate` | 0.01 | Low rate to prevent overfitting on noisy market data. |
| `max_depth` | 6 | Constrains tree depth to control model complexity. |
| `l2_regularization` | 1.0 | Reduces sensitivity to noise (crucial in finance). |

## 📈 Strategy Optimization (Finding $k$)
The model predictions are converted into trading positions. The optimizer searches for a scalar $k$ that maximizes a **Custom Sharpe Ratio**.

**The Position Formula:**
$$\text{Position} = \text{clip}(1.0 + (\text{Prediction} \times k), \ 0, \ 2)$$

**The Optimization Metric:**
The custom score penalizes volatility and market underperformance:
1.  **Base Score:** Standard Sharpe Ratio.
2.  **Volatility Penalty:** Applied if Strategy Volatility > $1.2 \times$ Market Volatility.
3.  **Return Penalty:** Applied if Strategy Returns < Market Returns.

*Grid Search Range for $k$:* `[0, 10, 30, 50, 80, 100, 150, 200]`

## 🚀 Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn
