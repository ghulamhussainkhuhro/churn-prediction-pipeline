#  Churn Prediction Pipeline

This project implements an end-to-end machine learning pipeline to predict customer churn using the [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn){:target="_blank"}. The entire workflow is built with **Scikit-learn Pipeline API** and includes preprocessing, training, hyperparameter tuning, model saving, and testing.

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

---

* **Accuracy**: 0.8038  
* **F1 Score**: 0.6080  

> The initial pipeline demonstrated strong performance. Hyperparameter tuning yielded similar results, confirming the model's robustness.

---

## 🔁 Reproducibility

```bash
# 1. Clone the repository
git clone https://github.com/ghulamhussainkhuhro/churn-prediction-pipeline.git
cd churn-prediction-pipeline

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt

# 3. Train the model
python pipeline/train.py

# 4. Perform hyperparameter tuning
python pipeline/tune.py

# 5. Load and test the pipeline
python test_loaded_pipeline.py
python test_tuned_pipeline.py

## 📘 Key Learnings

- Constructing ML pipelines using `Pipeline` and `ColumnTransformer`  
- Performing hyperparameter tuning with `GridSearchCV`  
- Persisting models with `joblib` for reuse  
- Evaluating models using **Accuracy** and **F1 Score**  

---

## 📬 Contact

For questions or suggestions, feel free to connect on [LinkedIn](https://www.linkedin.com/ghulamhussainkhuhro).

---