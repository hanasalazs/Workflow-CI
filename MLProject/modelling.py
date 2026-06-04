import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_baseline():
    mlflow.autolog()
    
    data_path = os.path.join("MLProject", "preprocessing", "student_performance_preprocessed.csv")
    
    if not os.path.exists(data_path):
        data_path = os.path.join("preprocessing", "student_performance_preprocessed.csv")

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        X = df.drop('G3', axis=1)
        y = df['G3']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run(run_name="LinearRegression_Baseline_Hana"):
            model = LinearRegression()
            model.fit(X_train, y_train)
            print("Baseline model selesai dilatih secara lokal.")
    else:
        print(f"Error: Dataset tidak ditemukan di path {data_path}")

if __name__ == "__main__":
    train_baseline()