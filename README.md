# Credit Risk Predictor 🏦

An end-to-end machine learning project that predicts the **probability of loan default** for retail banking applicants, and translates it into an interpretable **credit score (300–900)** and **risk rating**.

🔗 **Live Demo:** [ml-credit-risk-predict.streamlit.app](https://ml-credit-risk-predict.streamlit.app/)  
📁 **Repo:** [github.com/shafiimran/ml_credit_risk_predictor](https://github.com/shafiimran/ml_credit_risk_predictor)

---

## Overview

Lenders need to assess borrower risk before sanctioning a loan. This project builds a binary classification model trained on **50,000 applicants** across 3 data sources — customer demographics, loan details, and credit bureau data — to predict default probability and surface it as an actionable credit scorecard.

---

## Project Pipeline

### 1. Data Preparation
- Merged 3 datasets (customers, loans, bureau data) on customer ID
- Imputed missing `residence_type` values using mode
- Removed outliers using domain business rules:
  - Processing fee must be < 3% of loan amount
  - GST must not exceed 20% of loan amount
  - Net disbursement must not exceed loan amount
- Fixed data entry errors (e.g. `Personaal` → `Personal` in loan purpose)

### 2. Exploratory Data Analysis
- Plotted KDE distributions for all continuous features split by default status
- Key findings:
  - `loan_tenure_months`, `delinquent_months`, `total_dpd`, and `credit_utilization_ratio` were the strongest default signals
  - `loan_amount` and `income` individually showed weak signal, but their ratio (Loan-to-Income) was highly predictive — motivating feature engineering

### 3. Feature Engineering

| Feature | Formula |
|---|---|
| Loan-to-Income Ratio | `loan_amount / income` |
| Delinquency Ratio | `(delinquent_months × 100) / total_loan_months` |
| Avg DPD per Delinquency | `total_dpd / delinquent_months` (0 if no delinquency) |

### 4. Feature Selection
- **VIF Analysis** — dropped multicollinear features: `sanction_amount`, `processing_fee`, `gst`, `net_disbursement`, `principal_outstanding`
- **Information Value (IV)** — retained only features with IV > 0.02
- **Encoding** — one-hot encoding with `drop_first=True`

### 5. Modelling

| Attempt | Model | Imbalance Handling | Tuning |
|---|---|---|---|
| 1 | Logistic Regression, Random Forest, XGBoost | None | RandomizedSearchCV |
| 2 | Logistic Regression, XGBoost | Random Undersampling | — |
| 3 | Logistic Regression | SMOTETomek | Optuna |
| 4 | XGBoost | SMOTETomek | Optuna |

**Final Model:** Logistic Regression + SMOTETomek + Optuna hyperparameter tuning  
Selected for strong performance and interpretability (model coefficients as feature importance).

### 6. Model Evaluation

**Classification Report — Final Model (Logistic Regression + SMOTETomek + Optuna)**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — Non-Default | 0.99 | 0.93 | 0.96 | 11,423 |
| 1 — Default | 0.55 | 0.94 | 0.70 | 1,074 |
| Accuracy | | | **0.93** | 12,497 |
| Macro Avg | 0.77 | 0.94 | 0.83 | 12,497 |
| Weighted Avg | 0.96 | 0.93 | 0.94 | 12,497 |

> The model achieves a **recall of 0.94 on defaulters** — correctly flagging 94% of
> actual defaults, which is the priority metric in credit risk. High precision on
> non-defaults (0.99) ensures low-risk applicants are not wrongly rejected.

**Rank-Ordering & Discrimination Metrics**

| Metric | Value |
|---|---|
| AUC | **0.98** |
| Gini Coefficient | **0.96** |
| KS Statistic | **85.98%** (at Decile 8) |

- KS in top 3 deciles and above 40 — meets industry benchmark for a strong credit risk model
- Decile 9 captures **72% of all defaulters**, confirming strong rank ordering
- Top 2 deciles cumulatively capture **98.6% of all defaulters**

### 7. Credit Scorecard

Default probability is mapped to a credit score between 300–900:

```
Credit Score = 300 + (1 - default_probability) × 600
```

| Score Range | Rating |
|---|---|
| 750 – 900 | Excellent ✅ |
| 650 – 749 | Good 🟡 |
| 500 – 649 | Average ⚠️ |
| 300 – 499 | Poor ❌ |

---

## Streamlit App

Interactive web app where users input applicant details and receive:
- **Default Probability** — likelihood of the applicant defaulting
- **Credit Score** — scaled between 300 and 900
- **Risk Rating** — Poor / Average / Good / Excellent

### App Inputs

| Section | Fields |
|---|---|
| Applicant Details | Age, Income, Residence Type, Credit Utilization Ratio, Open Loan Accounts |
| Loan Information | Loan Amount, Loan Purpose, Loan Type |
| Delinquency Details | Delinquent Months, Loan Tenure, Total DPD |
| Derived (read-only) | Loan-to-Income Ratio, Delinquency Ratio, Avg DPD per Delinquency |

---

## Repo Structure

```
ml_credit_risk_predictor/
├── artifacts/
│   └── model_data.joblib      # Saved model, scaler, features
├── main.py                    # Streamlit UI
├── prediction_helper.py       # Model loading, feature prep, prediction logic
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/shafiimran/ml_credit_risk_predictor.git
cd ml_credit_risk_predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run main.py
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Data & EDA | Pandas, NumPy, Matplotlib, Seaborn |
| ML & Resampling | Scikit-learn, XGBoost, Imbalanced-learn |
| Hyperparameter Tuning | Optuna, RandomizedSearchCV |
| Deployment | Streamlit, Joblib |

---

## Author

**Shafi Imran**  
Aspiring Data Scientist | Business & Technology Undergrad  
[GitHub](https://github.com/shafiimran) · [LinkedIn](https://www.linkedin.com/in/shafiimran)
