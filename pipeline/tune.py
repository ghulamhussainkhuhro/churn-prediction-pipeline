# pipeline/tune.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from pipeline.preprocess import Preprocessor

# 1. Load and clean data
df = pd.read_csv("data/telco_churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Define pipeline and grid
preprocessor = Preprocessor()

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

param_grid = [
    {
        'model': [LogisticRegression(max_iter=1000)],
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l2'],
        'model__solver': ['liblinear', 'lbfgs']
    },
    {
        'model': [RandomForestClassifier(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [5, 10, None],
        'model__min_samples_split': [2, 5]
    }
]
# 3. Grid search
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

# 4. Best model
best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)

# 5. Evaluate on test
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 6. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/churn_pipeline_tuned.joblib")

# 7. Log metrics
os.makedirs("results", exist_ok=True)
with open("results/tuned_metrics.txt", "w") as f:
    f.write(f"Best Params: {grid.best_params_}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print("✅ Tuned pipeline saved to models/churn_pipeline_tuned.joblib")
