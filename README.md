# 📊 Churn Prediction Pipeline

A reusable and production-ready machine learning pipeline to predict whether a customer will churn, using structured telco data.

---

## 🎯 Objective

Build a modular and maintainable ML pipeline capable of predicting customer churn with strong accuracy and F1 score, using scikit-learn tools and best practices.

---

## 🛠️ Tech Stack

- **Python**
- **scikit-learn**
- **pandas**, **numpy**
- **matplotlib**, **seaborn**
- **joblib**

---

## 🗂️ Project Structure

```
churn-prediction-pipeline/
│
├── data/                   # Raw dataset
│   └── telco_churn.csv
│
├── models/                 # Saved pipelines
│   ├── churn_pipeline.joblib
│   └── churn_pipeline_tuned.joblib
│
├── notebooks/              # Notebooks for EDA & final pipeline
│   ├── EDA.ipynb
│   └── final_pipeline.ipynb
│
├── pipeline/               # Core pipeline logic
│   ├── preprocess.py
│   ├── train.py
│   └── tune.py
│
├── results/                # Evaluation results
│   ├── churn_distribution.png
│   ├── metrics.txt
│   └── tuned_metrics.txt
│
├── test_preprocess.ipynb       # Testing pipeline steps
├── test_loaded_pipeline.py     # Load & test basic pipeline
├── test_tuned_pipeline.py      # Load & test tuned pipeline
├── requirements.txt
└── README.md
```

---

## 📈 Results

🔹 **Base Pipeline**  
- Accuracy: `0.8038`  
- F1 Score: `0.6080`  

🔹 **Tuned Pipeline** *(Logistic Regression + GridSearchCV)*  
Best Parameters:
```python
{
  'model': LogisticRegression(max_iter=1000),
  'model__C': 1,
  'model__penalty': 'l2',
  'model__solver': 'liblinear'
}
```

> The initial pipeline demonstrated strong performance. Hyperparameter tuning yielded similar results, confirming the model's robustness.

---

## 🔁 Reproducibility

### 1. Clone the repository
```bash
git clone https://github.com/ghulamhussainkhuhro/churn-prediction-pipeline.git
cd churn-prediction-pipeline
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 3. Train the model
```bash
python pipeline/train.py
```

### 4. Perform hyperparameter tuning
```bash
python pipeline/tune.py
```

### 5. Load and test the pipeline
```bash
python test_loaded_pipeline.py
python test_tuned_pipeline.py
```

---

## 📘 Key Learnings

- Building ML pipelines using `Pipeline` and `ColumnTransformer`  
- Hyperparameter tuning with `GridSearchCV`  
- Persisting trained models with `joblib`  
- Evaluating models using **Accuracy** and **F1 Score**  
- Writing modular and testable pipeline code  

---

## 🔗 Dataset

[Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## 📬 Contact

For questions or suggestions, feel free to connect on [LinkedIn](https://www.linkedin.com/in/ghulamhussainkhuhro).
