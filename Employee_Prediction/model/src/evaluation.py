import sklearn.metrics as metrics
import numpy as np
from config import FP_COST, FN_COST
import pandas as pd

def evaluate_classification(y1_test, y1_pred):
    accuracy = metrics.accuracy_score(y1_test, y1_pred)
    precision = metrics.precision_score(y1_test, y1_pred, pos_label=1)
    classification_report = metrics.classification_report(y1_test, y1_pred)
    recall = metrics.recall_score(y1_test, y1_pred, pos_label=1)
    f1 = metrics.f1_score(y1_test, y1_pred, pos_label=1)

    # Confusion Matrix
    tn, fp, fn, tp = metrics.confusion_matrix(y1_test, y1_pred, labels=[0, 1]).ravel()

    # Cost Calculation
    total_cost = (fp * FP_COST) + (fn * FN_COST)
    
    evaluation_results = {
        'ACCURACY_SCORE': accuracy,
        'PRECISION_SCORE': precision,
        'RECALL_SCORE': recall,
        'F1_SCORE': f1,
        'FALSE_POSITIVES': int(fp),
        'FALSE_NEGATIVES': int(fn),
        'TOTAL_COST': total_cost,
        'CLASSIFICATION_REPORT': classification_report
    }
    print("Evaluation Results for Random Forest Classifier:", pd.DataFrame([evaluation_results], columns=evaluation_results.keys()))
    print("✅ Classification Evaluation Metrics Computed Successfully!")
    return evaluation_results

def evaluate_regression(y2_test: np.ndarray, y2_pred: np.ndarray) -> dict:
    mse = metrics.mean_squared_error(y2_test, y2_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y2_test, y2_pred)
    r2 = metrics.r2_score(y2_test, y2_pred)

    evaluation_results = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2
    }
    print("Evaluation Results for Linear Regression:", pd.DataFrame([evaluation_results], columns=evaluation_results.keys()))
    print("✅ Regression Evaluation Metrics Computed Successfully!")
    return evaluation_results