"""
Bank Loan Case Study — Analysis Script
=======================================
Risk assessment and EDA to predict loan approval outcomes.
Uses Logistic Regression and Random Forest classification.

Usage:
    python scripts/analysis.py

Output:
    - EDA summary printed to console
    - Model performance metrics printed
    - 4 charts saved to outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import os
import warnings
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'bank_loan_data.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 130

# ── 1. Load & Validate ─────────────────────────────────────────────────────────
print("=" * 55)
print("  BANK LOAN CASE STUDY")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Loaded {len(df)} rows × {len(df.columns)} columns")
print(f"   Missing   : {df.isnull().sum().sum()} null values")
print(f"   Duplicates: {df.duplicated().sum()} duplicate rows")
print(f"   Approval Rate: {df['Loan_Status'].mean()*100:.1f}%")

# ── 2. EDA ─────────────────────────────────────────────────────────────────────
print("\n── EDA SUMMARY ──────────────────────────────────────")
print(df.describe().round(2))

print("\n── APPROVAL RATE BY SEGMENT ─────────────────────────")

# Credit score segments
bins   = [300, 580, 670, 740, 850]
labels = ['Poor (300-579)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740-850)']
df['Credit_Segment'] = pd.cut(df['Credit_Score'], bins=bins, labels=labels)

risk = df.groupby('Credit_Segment', observed=True)['Loan_Status'].agg(
    Total='count', Approved='sum',
    Approval_Rate=lambda x: f"{x.mean()*100:.1f}%"
)
print("\nBy Credit Score Segment:")
print(risk.to_string())

print("\nBy Education:")
print(df.groupby('Education')['Loan_Status'].mean().mul(100).round(1).to_string())

print("\nBy Property Area:")
print(df.groupby('Property_Area')['Loan_Status'].mean().mul(100).round(1).to_string())

print("\nBy Self Employed:")
print(df.groupby('Self_Employed')['Loan_Status'].mean().mul(100).round(1).to_string())

# ── 3. Prepare for ML ──────────────────────────────────────────────────────────
print("\n── MACHINE LEARNING ─────────────────────────────────")
df_model = df.copy()
le = LabelEncoder()
for col in ['Education', 'Self_Employed', 'Property_Area', 'Credit_Segment']:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

features = ['Age', 'Income', 'Loan_Amount', 'Credit_Score',
            'Employment_Years', 'Existing_Loans',
            'Education', 'Self_Employed', 'Property_Area']
X = df_model[features]
y = df_model['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===")
print(f"Accuracy : {accuracy_score(y_test, lr_preds)*100:.1f}%")
print(f"ROC-AUC  : {roc_auc_score(y_test, lr_proba):.3f}")
print(classification_report(y_test, lr_preds, target_names=['Rejected','Approved']))

print("=== Random Forest ===")
print(f"Accuracy : {accuracy_score(y_test, rf_preds)*100:.1f}%")
print(f"ROC-AUC  : {roc_auc_score(y_test, rf_proba):.3f}")
print(classification_report(y_test, rf_preds, target_names=['Rejected','Approved']))

# ── 4. Charts ──────────────────────────────────────────────────────────────────
print("\n── GENERATING CHARTS ────────────────────────────────")

# Chart 1 — EDA overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bank Loan — Exploratory Data Analysis', fontsize=14, fontweight='bold')

# Approval distribution
counts = df['Loan_Status'].value_counts()
axes[0,0].bar(['Rejected', 'Approved'], counts.values,
              color=['#F44336', '#4CAF50'], edgecolor='white')
axes[0,0].set_title('Loan Approval Distribution')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0,0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Credit score by status
sns.boxplot(data=df, x='Loan_Status', y='Credit_Score',
            palette=['#F44336', '#4CAF50'], ax=axes[0,1])
axes[0,1].set_title('Credit Score by Loan Status')
axes[0,1].set_xticklabels(['Rejected', 'Approved'])
axes[0,1].set_xlabel('')

# Income distribution
sns.histplot(data=df, x='Income', hue='Loan_Status', bins=30,
             kde=True, ax=axes[1,0], palette=['#F44336', '#4CAF50'])
axes[1,0].set_title('Income Distribution by Status')
axes[1,0].set_xlabel('Income (₹)')

# Approval rate by credit segment
seg_rate = df.groupby('Credit_Segment', observed=True)['Loan_Status'].mean().mul(100)
seg_rate.plot.bar(ax=axes[1,1], color=['#F44336','#FF9800','#2196F3','#4CAF50'],
                  edgecolor='white', rot=15)
axes[1,1].set_title('Approval Rate by Credit Score Segment')
axes[1,1].set_ylabel('Approval Rate (%)')
axes[1,1].axhline(df['Loan_Status'].mean()*100, color='black',
                  linestyle='--', linewidth=1, label='Overall avg')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_01_eda_overview.png'), bbox_inches='tight')
plt.close()
print("  ✅ chart_01_eda_overview.png")

# Chart 2 — Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')

for ax, preds, title in zip(axes,
    [lr_preds, rf_preds],
    ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Rejected','Approved'],
                yticklabels=['Rejected','Approved'])
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_02_confusion_matrices.png'), bbox_inches='tight')
plt.close()
print("  ✅ chart_02_confusion_matrices.png")

# Chart 3 — ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
for proba, name, color in [
    (lr_proba, 'Logistic Regression', '#2196F3'),
    (rf_proba, 'Random Forest', '#4CAF50')]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})', color=color, linewidth=2)

ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Random guess')
ax.fill_between(*roc_curve(y_test, rf_proba)[:2], alpha=0.08, color='#4CAF50')
ax.set_title('ROC Curve — Model Comparison', fontsize=13, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_03_roc_curve.png'), bbox_inches='tight')
plt.close()
print("  ✅ chart_03_roc_curve.png")

# Chart 4 — Feature Importance
fig, ax = plt.subplots(figsize=(9, 5))
fi = pd.Series(rf.feature_importances_, index=features).sort_values()
colors = ['#d32f2f' if v == fi.max() else '#2E75B6' for v in fi.values]
fi.plot.barh(ax=ax, color=colors)
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.xaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_04_feature_importance.png'), bbox_inches='tight')
plt.close()
print("  ✅ chart_04_feature_importance.png")

print("\n✅ All outputs saved to /outputs/")
print("=" * 55)
print("  ANALYSIS COMPLETE")
print("=" * 55)
