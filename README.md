# Credit Card Transaction Fraud Detection

Fraud detection on 6.3M+ imbalanced financial transactions using Logistic Regression, Random Forest, and XGBoost, with SHAP-based explainability.

## Overview

This project builds a binary classification pipeline to detect fraudulent transactions in a highly imbalanced mobile-money / financial transactions dataset (6.36M rows, ~0.13% fraud). It covers exploratory data analysis, feature engineering, model training and comparison, feature importance analysis, SHAP explainability, and a simple fraud-scoring function.

## Dataset

The dataset is a PaySim-style synthetic financial transactions dataset with the following columns:

| Column | Description |
|---|---|
| `step` | Time unit (1 step = 1 hour) |
| `type` | Transaction type (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `CASH_IN`, `DEBIT`) |
| `amount` | Transaction amount |
| `nameOrig` | Origin account ID |
| `oldbalanceOrg` / `newbalanceOrig` | Origin account balance before/after transaction |
| `nameDest` | Destination account ID |
| `oldbalanceDest` / `newbalanceDest` | Destination account balance before/after transaction |
| `isFraud` | Target — whether the transaction is fraudulent |
| `isFlaggedFraud` | Dataset's own rule-based fraud flag |

> The raw CSV is not included in this repo due to size. See [Data](#data) below for where to get it.

## Pipeline

1. **Data loading & initial inspection** — shape, dtypes, nulls, duplicates
2. **Exploratory Data Analysis (EDA)** — fraud rate by transaction type, amount distributions, fraud volume over time
3. **Data cleaning** — duplicate removal, sanity checks on balances and amounts
4. **Feature engineering** — log-transformed amount, origin/destination transaction counts, balance-change deltas, amount discrepancies, hour/day derived from `step`
5. **Train/test split & preprocessing** — 80/20 stratified split, class weighting for imbalance
6. **Model training & evaluation** — Logistic Regression, Random Forest, XGBoost
7. **Feature importance analysis** — comparison across Random Forest and XGBoost
8. **Model explainability with SHAP** — summary, force, and waterfall plots
9. **Fraud prediction function** — a `predict_fraud()` helper to score a single transaction

## Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.006 | 0.863 | 0.012 | 0.934 | 0.036 |
| Random Forest | 0.729 | 0.562 | 0.635 | 0.917 | 0.609 |
| XGBoost | **0.911** | 0.497 | **0.643** | **0.979** | **0.640** |

**Key takeaways:**
- Fraud is concentrated entirely in `TRANSFER` and `CASH_OUT` transactions — `PAYMENT`, `CASH_IN`, and `DEBIT` have zero fraud cases in this dataset.
- The dataset's built-in `isFlaggedFraud` rule catches only 16 of 8,213 fraud cases and is not useful on its own.
- **PR-AUC matters more than ROC-AUC** on this kind of rare-event problem — Logistic Regression's high ROC-AUC (0.93) is misleading given its near-zero precision.
- **XGBoost is the best-performing model**, with the highest precision and PR-AUC. It still misses about half of actual fraud, so a production system might lower the decision threshold to trade some precision for recall, since missed fraud is typically costlier than a false alarm.
- Feature importance is broadly consistent across models: `destination_transaction_count`, `amount`/`log_amount`, and transaction `type` (especially `TRANSFER`) are the strongest predictors.

## Project Structure

```
.
├── credit_card_fraud_detection.ipynb   # Main analysis notebook
├── requirements.txt                     # Python dependencies
├── README.md
└── .gitignore
```

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Data

This project uses a PaySim-style synthetic financial transactions dataset. Place the CSV file (expected filename: `credit card transactions.csv`) in the project root before running the notebook. A commonly used public version of this dataset is available on Kaggle ("PaySim: Synthetic Financial Datasets For Fraud Detection").

## Usage

Open and run `credit_card_fraud_detection.ipynb` in Jupyter:

```bash
jupyter notebook credit_card_fraud_detection.ipynb
```

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- XGBoost
- SHAP
- matplotlib

## License

MIT — see [LICENSE](LICENSE) for details.
