# test_loaded_pipeline.py

import joblib
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load data (sample)
df = pd.read_csv("data/telco_churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Churn'])
y = df['Churn']

# Load pipeline
pipeline = joblib.load("models/churn_pipeline_tuned.joblib")

# Make prediction
print("Prediction for first 5 rows:")
print(pipeline.predict(X.head()))
