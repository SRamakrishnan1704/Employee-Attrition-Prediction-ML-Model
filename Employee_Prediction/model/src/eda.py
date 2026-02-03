import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path: str) -> pd.DataFrame:
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        print(f"✅ Data loaded: {data.shape}")
        return data 
    else:
        raise FileNotFoundError(f"❌ File not found: {file_path}")

def analyze_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA Step 1: Data quality analysis
    Returns original dataframe unchanged for downstream processing
    """
    print("\n" + "="*60)
    print("🔍 DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Shape & basic info
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {len(df.columns)}")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"📊 Duplicate rows: {duplicates}")
    
    # Null values
    null_total = df.isnull().sum().sum()
    if null_total == 0:
        print("✅ No null values")
    else:
        print(f"❌ {null_total} null values found")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Target distribution (Attrition)
    if 'Attrition' in df.columns:
        print(f"\n🎯 Attrition distribution:")
        print(df['Attrition'].value_counts(normalize=True).round(3))
    
    # Performance target
    if 'PerformanceRating' in df.columns:
        print(f"\n📈 PerformanceRating distribution:")
        print(df['PerformanceRating'].value_counts(normalize=True).round(3))
    
    # Data types overview
    print(f"\n🔬 Data types:")
    print(df.dtypes.value_counts())
    
    return df