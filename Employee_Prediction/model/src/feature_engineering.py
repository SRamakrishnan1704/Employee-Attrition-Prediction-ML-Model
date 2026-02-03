import numpy as np
import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Dropping the unwanted columns
    df.drop(columns=['Age','DailyRate','Department','Education', 'EducationField','EmployeeCount',
                     'EmployeeNumber','HourlyRate','JobLevel','JobRole','MonthlyRate',
                     'Over18','PercentSalaryHike','RelationshipSatisfaction',
                     'StandardHours','TrainingTimesLastYear','YearsSinceLastPromotion',
                     'YearsWithCurrManager'], inplace=True)
    return df