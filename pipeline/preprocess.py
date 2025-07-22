# pipeline/preprocess.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline = None
        self.categorical_cols = []
        self.numerical_cols = []

    def fit(self, X, y=None):
        X = X.copy()

        # Drop customerID if it exists
        if 'customerID' in X.columns:
            X = X.drop('customerID', axis=1)

        # Fix TotalCharges conversion
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

        # Identify columns
        self.categorical_cols = X.select_dtypes(include=['object']).drop(columns=['Churn'], errors='ignore').columns.tolist()
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Create transformers
        transformers = [
            ('num', StandardScaler(), self.numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_cols)
        ]

        self.pipeline = ColumnTransformer(transformers)
        self.pipeline.fit(X)

        return self

    def transform(self, X):
        X = X.copy()

        # Drop customerID if it exists
        if 'customerID' in X.columns:
            X = X.drop('customerID', axis=1)

        # Fix TotalCharges conversion
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

        return self.pipeline.transform(X)

    def get_feature_names(self):
        cat_feature_names = self.pipeline.named_transformers_['cat'].get_feature_names_out(self.categorical_cols)
        return np.concatenate([self.numerical_cols, cat_feature_names])
