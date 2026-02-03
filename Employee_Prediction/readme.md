README.md for IBM Employee Attrition Prediction Project

This project analyzes the IBM HR dataset to predict employee attrition (Yes/No via Random Forest) and performance ratings (via Linear Regression), deployed as a Streamlit dashboard. It uses 15 key features from the dataset and includes top/bottom performer visualizations.

Project Overview
Built for HR analytics, this ML pipeline preprocesses data, trains models, saves artifacts (pipelines and features), and serves predictions interactively. Key components include EDA, feature engineering with scaling/encoding, and a production-ready Streamlit app at D:/Ramakrishnan S/Guvi/Visual studio/My Project foler/Employee_Prediction/app/.

Features
Dual Predictions: Attrition risk and performance score on one page.
​

Visualizations: Top 10 performers, bottom 10 at-risk employees, performance distribution.
​

Interactive Inputs: All 15 features (Age, BusinessTravel, etc.) via dropdowns/sliders.

Model Pipelines: Preprocessing + Random Forest/Linear Regression, saved as .pkl files.

Tech Stack
Component	Tools
ML Models	scikit-learn (Random Forest, Linear Regression), joblib[pickle]
App Framework	Streamlit
Data Handling	pandas, numpy
Environment	Python, VS Code (Windows) 
​
Quick Start
Navigate to app folder: cd "D:/Ramakrishnan S/Guvi/Visual studio/My Project foler/Employee_Prediction/app/"
​

Install dependencies: pip install streamlit pandas scikit-learn joblib numpy

Run app: streamlit run app.py

Access dashboard: http://localhost:8501

Verify models exist: dir *.pkl (should show attrition_pipeline.pkl, performance_pipeline.pkl, features.pkl).
​

Employee_Prediction/
├── README.md                 ✅ Your documentation
├── config/                   # Configuration files
│   └── config.py
├── loader/                   # Data loading scripts
├── eda/                      # Exploratory data analysis
├── feature_engineering/      # Feature creation/transformation
├── preprocessing/            # Scaling, encoding, cleaning
├── modeling/                 # Model training code
├── evaluation/               # Metrics, validation, testing
├── train_pipeline.py         # Main training orchestrator
└── App.py                    # Streamlit dashboard

Training Models
Run python train_pipeline.py from project root to retrain/save models (uses RANDOM_STATE=42, TEST_SIZE=0.2). Ensure dataset is in data/.
​

Usage
Enter employee details (15 features).

Get predictions: Attrition probability, performance rating.

View tables/charts for insights.

Future Improvements
Add hyperparameter tuning (GridSearchCV).

Deploy to cloud (Streamlit Cloud/Hugging Face).

Integrate SQL/Power BI for live data.

Project ready !!.
​