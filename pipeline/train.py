# pipeline/train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from pipeline.preprocess import Preprocessor



# 1. Load data
df = pd.read_csv("data/telco_churn.csv")

# 2. Clean up target
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. Split features & target
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Choose model
model = LogisticRegression(max_iter=1000)  # Try RandomForestClassifier() here too

# 5. Build full pipeline
preprocessor = Preprocessor()
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', model)
])

# 6. Train
pipeline.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 8. Save metrics
os.makedirs("results", exist_ok=True)
with open("results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

# 9. Save trained pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/churn_pipeline.joblib")
print("✅ Pipeline saved to models/churn_pipeline.joblib")
