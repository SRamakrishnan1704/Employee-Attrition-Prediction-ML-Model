# Models building
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib as jb
from config import RANDOM_STATE, FP_COST, FN_COST
from preprocessing import preprocess_data
from loader import load_data
from feature_engineering import feature_engineering
from config import file_path, TARGET_COLUMN_1, TARGET_COLUMN_2

def train_randomforest_classifier(X_train, y1_train) -> RandomForestClassifier:
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE,min_samples_split=3,min_samples_leaf=1,class_weight='balanced')
    rf_model.fit(X_train, y1_train)
    print("✅ Random Forest Classifier trained successfully!")
    return rf_model

def train_linear_regression(X_train, y2_train) -> LinearRegression:
    lr_model = LinearRegression()
    lr_model.fit(X_train, y2_train)
    print("✅ Linear Regression model trained successfully!")
    return lr_model

# Save Model
def save_model(model, file_path: str) -> None:
    jb.dump(model, file_path)
    print(f"✅ Model saved to {file_path}")

