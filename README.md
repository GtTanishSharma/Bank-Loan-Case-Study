# Bank Loan Case Study

Took a dataset of 1000 loan applications and tried to figure out what separates approved loans from rejected ones. Then built two models to predict approval outcomes.

---

## Files

- `data/bank_loan_data.csv` — the dataset (1000 applications)
- `scripts/analysis.py` — full code
- `outputs/` — all 4 charts

---

## The Data

1000 loan applications with these columns:

| Column | What it means |
|---|---|
| Age | Applicant age |
| Income | Annual income in ₹ |
| Loan_Amount | Amount applied for in ₹ |
| Credit_Score | Score between 300–850 |
| Employment_Years | Years at current job |
| Existing_Loans | Number of active loans |
| Education | Graduate or Not Graduate |
| Self_Employed | Yes or No |
| Property_Area | Urban, Rural, Semiurban |
| Loan_Status | 1 = Approved, 0 = Rejected |

Overall approval rate in the dataset: **76.8%**

---

## What I Did

Cleaned the data, checked for nulls and duplicates, then did a risk segmentation analysis — breaking applicants into groups by credit score to see how approval rates changed.

After EDA, encoded the categorical columns and trained two classification models — Logistic Regression and Random Forest — on an 80/20 train-test split.

---

## What I Found

**Credit score is everything**

The difference is dramatic:

| Credit Segment | Approval Rate |
|---|---|
| Poor (300–579) | 58.5% |
| Fair (580–669) | 90.5% |
| Good (670–739) | 96.3% |
| Excellent (740–850) | 98.0% |

Going from Poor to Fair credit nearly doubles your approval chances. That's the single biggest lever an applicant has.

**Income and Employment Years matter but less than expected**

Both models ranked Credit Score as the most important feature by a significant margin. Income came second, Employment Years third. Loan amount had surprisingly low importance — banks seem to care more about who you are than how much you're asking for.

**Education and Self Employment barely mattered**

Approval rates were almost identical between graduates and non-graduates (76% vs 77.6%). Self-employed applicants were actually approved slightly more often. These are commonly assumed to be big factors but the data doesn't support that here.

**Both models performed similarly**

Logistic Regression: 88.5% accuracy, AUC 0.955
Random Forest: 88.5% accuracy, AUC 0.949

Logistic Regression had a slightly better AUC despite being the simpler model — which tells you the relationship between features and approval is fairly linear. The Random Forest didn't find complex patterns that improved on it.

---

## Model Performance

```
=== Logistic Regression ===
Accuracy : 88.5%
ROC-AUC  : 0.955

=== Random Forest ===
Accuracy : 88.5%
ROC-AUC  : 0.949
```

---

## Charts

### EDA Overview
![EDA](outputs/chart_01_eda_overview.png)

### Confusion Matrices
![Confusion](outputs/chart_02_confusion_matrices.png)

### ROC Curve
![ROC](outputs/chart_03_roc_curve.png)

### Feature Importance
![Features](outputs/chart_04_feature_importance.png)

---

## How to Run

```
pip install -r requirements.txt
python scripts/analysis.py
```

---

**Tanish Sharma**
[LinkedIn](https://linkedin.com/in/grt-tanish) · Tanishshr1234@gmail.com
