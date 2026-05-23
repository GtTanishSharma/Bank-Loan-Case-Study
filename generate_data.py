import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 1000

age              = np.random.randint(21, 65, n)
income           = np.random.randint(20000, 200000, n)
loan_amount      = np.random.randint(5000, 50000, n)
credit_score     = np.random.randint(300, 850, n)
employment_years = np.random.randint(0, 30, n)
existing_loans   = np.random.randint(0, 5, n)
education        = np.random.choice(['Graduate', 'Not Graduate'], n)
self_employed    = np.random.choice(['Yes', 'No'], n, p=[0.2, 0.8])
property_area    = np.random.choice(['Urban', 'Rural', 'Semiurban'], n)

approval_score = (
    (credit_score - 300) / 550 * 0.4 +
    income / 200000 * 0.3 +
    employment_years / 30 * 0.2 +
    (1 - loan_amount / 50000) * 0.1 +
    np.random.uniform(0, 0.2, n)
)
loan_status = (approval_score > 0.5).astype(int)

df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Loan_Amount': loan_amount,
    'Credit_Score': credit_score,
    'Employment_Years': employment_years,
    'Existing_Loans': existing_loans,
    'Education': education,
    'Self_Employed': self_employed,
    'Property_Area': property_area,
    'Loan_Status': loan_status
})

out = os.path.join(os.path.dirname(__file__), 'bank_loan_data.csv')
df.to_csv(out, index=False)
print(f"Generated {len(df)} rows → {out}")
print(f"Approval Rate: {df['Loan_Status'].mean()*100:.1f}%")
