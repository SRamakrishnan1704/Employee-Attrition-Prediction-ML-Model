import numpy as np
import pandas as pd
import joblib as jb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LinearRegression

def preprocess_data(df):
    
    unified_features = ['BusinessTravel', 'DistanceFromHome', 'EnvironmentSatisfaction', 
                       'Gender', 'JobInvolvement', 'JobSatisfaction', 'MaritalStatus', 
                       'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 
                       'StockOptionLevel', 'TotalWorkingYears', 'WorkLifeBalance', 
                       'YearsAtCompany', 'YearsInCurrentRole']
    
    X = df[unified_features]
    y1 = (df['Attrition'] == 'Yes').astype(int)  # Attrition
    y2 = df['PerformanceRating']  # Performance
    
    # Single split for consistency
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        X, y1, y2, test_size=0.2, random_state=42, stratify=y1)
    
    return X_train, X_test, y1_train, y1_test, y2_train, y2_test, unified_features

# Update train_and_save_models
def train_and_save_models(X_train, X_test, y1_train, y1_test, y2_train, y2_test, features):
    # ✅ Split YOUR 15 features: 7 categorical + 8 numerical
    cat_features = ['BusinessTravel', 'EnvironmentSatisfaction', 'Gender', 
                   'JobInvolvement', 'JobSatisfaction', 'MaritalStatus', 'OverTime']
    num_features = ['DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 
                   'StockOptionLevel', 'TotalWorkingYears', 'WorkLifeBalance', 
                   'YearsAtCompany', 'YearsInCurrentRole']
    
    # Attrition pipeline (your top 5: OverTime, MonthlyIncome, etc.)
    attr_pipeline = ImbPipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])),
        ('smote_tomek', SMOTETomek(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    attr_pipeline.fit(X_train, y1_train)
    
    # Performance pipeline (same preprocessor)
    perf_pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])),
        ('regressor', LinearRegression())
    ])
    perf_pipeline.fit(X_train, y2_train)
    
    # ✅ Save everything
    jb.dump(attr_pipeline, 'attrition_pipeline.pkl')
    jb.dump(perf_pipeline, 'performance_pipeline.pkl')
    jb.dump(features, 'features.pkl')  # Single feature list!
    
    # Evaluate
    print(f"Attrition ROC-AUC: {attr_pipeline.score(X_test, y1_test):.3f}")
    print(f"Perf R²: {perf_pipeline.score(X_test, y2_test):.3f}")
    
    return attr_pipeline, perf_pipeline