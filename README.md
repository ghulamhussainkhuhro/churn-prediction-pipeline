#  Churn Prediction Pipeline

This project implements an end-to-end machine learning pipeline to predict customer churn using the [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn). The entire workflow is built with **Scikit-learn Pipeline API** and includes preprocessing, training, hyperparameter tuning, model saving, and testing.

---

##  Objective

Build a **reusable and production-ready machine learning pipeline** to predict whether a customer will churn, using structured telco data.

---

##  Tech Stack

- Python
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- joblib

---

##  Project Structure

```

churn-prediction-pipeline/
│
├── data/                   # Raw dataset
│   └── telco\_churn.csv
│
├── models/                 # Saved pipelines
│   ├── churn\_pipeline.joblib
│   └── churn\_pipeline\_tuned.joblib
│
├── notebooks/              # Notebooks for EDA & final pipeline
│   ├── EDA.ipynb
│   └── final\_pipeline.ipynb
│
├── pipeline/               # Core pipeline logic
│   ├── preprocess.py
│   ├── train.py
│   └── tune.py
│
├── results/                # Evaluation results
│   ├── churn\_distribution.png
│   ├── metrics.txt
│   └── tuned\_metrics.txt
│
├── test\_preprocess.ipynb   # Testing pipeline steps
├── test\_loaded\_pipeline.py # Load & test basic pipeline
├── test\_tuned\_pipeline.py  # Load & test tuned pipeline
├── requirements.txt
└── README.md

````

---

##  Results

### 🔹 Base Pipeline:
- **Accuracy**: 0.8038
- **F1 Score**: 0.6080

### 🔹 Tuned Pipeline (Logistic Regression + GridSearchCV):
- **Best Params**:  
  ```python
  {
    'model': LogisticRegression(max_iter=1000),
    'model__C': 1,
    'model__penalty': 'l2',
    'model__solver': 'liblinear'
  }
````

* **Accuracy**: 0.8038
* **F1 Score**: 0.6080

> Note: Hyperparameter tuning showed comparable performance, validating robustness of the initial pipeline.

---

##  Reproducibility

### 1. Clone the repo

```bash
git clone https://github.com/ghulamhussainkhuhro/churn-prediction-pipeline.git
cd churn-prediction-pipeline
```

### 2. Setup virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Run training

```bash
python pipeline/train.py
```

### 4. Run tuning

```bash
python pipeline/tune.py
```

### 5. Load & test pipeline

```bash
python test_loaded_pipeline.py
python test_tuned_pipeline.py
```

---

##  Key Learnings

* Building production-ready ML pipelines using `Pipeline`, `ColumnTransformer`
* Hyperparameter tuning with `GridSearchCV`
* Model export with `joblib` for reuse
* Evaluation using **Accuracy** and **F1 Score**

---

##  Contact

For queries or suggestions, reach out on [LinkedIn](https://www.linkedin.com).

