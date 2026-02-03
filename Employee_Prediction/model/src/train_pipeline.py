import pandas as pd
import numpy as np
import joblib as jb
import os
from config import *
from sklearn.ensemble import RandomForestClassifier
from loader import load_data
from eda import analyze_data_quality
from feature_engineering import feature_engineering
from preprocessing import preprocess_data, train_and_save_models 
from evaluation import evaluate_classification, evaluate_regression

def train_pipeline():
    print("🚀 COMPLETE EMPLOYEE ATTRITION TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load + EDA (feature_engineering keeps 15 features!)
    data = load_data("D:/Ramakrishnan S/Guvi/Visual studio/My Project foler/Employee_Prediction/data/Employee_Attrition.csv")
    print("📊 Data loaded:", data.shape)
    
    data = analyze_data_quality(data)
    data_fe = feature_engineering(data)  # Now keeps ALL 15 features!
    print("📊 After feature engineering:", data_fe.shape)
    
    # 2. Preprocess - FIXED UNPACKING (7 returns)
    X_train, X_test, y1_train, y1_test, y2_train, y2_test, features = preprocess_data(data_fe)
    print(f"✅ 15 features selected: {len(features)}")
    
    # 3. Train & Save - FIXED ARGS (7 args only)
    attr_pipeline, perf_pipeline = train_and_save_models(
        X_train, X_test, y1_train, y1_test, y2_train, y2_test, features
    )
    
    # 4. SIMPLIFIED EVALUATION (use pipeline directly)
    print("\n📊 PRODUCTION EVALUATION")
    print("-"*40)
    
    # Attrition (direct pipeline score - honest!)
    attr_accuracy = attr_pipeline.score(X_test, y1_test)
    print(f"✅ ATTRITION ACCURACY: {attr_accuracy:.3f}")
    
    # Performance
    perf_r2 = perf_pipeline.score(X_test, y2_test)
    print(f"✅ PERFORMANCE R²: {perf_r2:.3f}")
    
    print("\n🎉 PRODUCTION READY!")
    os.makedirs(APP_DIR, exist_ok=True)
    
    jb.dump(attr_pipeline, os.path.join(APP_DIR, 'attrition_pipeline.pkl'))
    jb.dump(perf_pipeline, os.path.join(APP_DIR, 'performance_pipeline.pkl'))
    jb.dump(features, os.path.join(APP_DIR, 'features.pkl'))
    
    print(f"✅ Files saved to APP FOLDER:")
    print(f"   • {os.path.join(APP_DIR, 'attrition_pipeline.pkl')}")
    print(f"   • {os.path.join(APP_DIR, 'performance_pipeline.pkl')}")
    print(f"   • {os.path.join(APP_DIR, 'features.pkl')}")
    
    print(f"✅ Streamlit needs these {len(features)} inputs:\n{features}")
if __name__ == "__main__":
    train_pipeline()