import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        data = pd.read_csv(file_path)
        print(f"✅ Loaded {data.shape[0]} rows, {data.shape[1]} columns")
        print("First 5 rows of the dataset:")
        print(data.head())
        return data
    else:
        raise FileNotFoundError(f"❌The file at {file_path} does not exist or is empty.")
