import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from preprocessing import run_preprocessing
from feature_engineering import run_feature_engineering
from validation import run_validation

def run_training_and_selection():
    print("[4/4] Starting MLflow Experiment Tracking & Model Selection (SQLite)...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Wine_Classification_MLOps")
    
    with open("feature_store/registry.json", "r") as f:
        registry = json.load(f)
    df = pd.read_parquet(registry["storage_path"])
    
    X = df[registry["features"]]
    y = df[registry["target"]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    best_f1 = -1.0
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        with mlflow.start_run(run_name=f"Wine_{name}"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")
            
            mlflow.log_param("model_type", name)
            mlflow.log_param("feature_count", len(registry["features"]))
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_artifact("feature_store/registry.json")
            mlflow.sklearn.log_model(model, artifact_path="model")
            
            print(f"Model: {name:20s} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_wine_model.pkl")
    print(f"\n[SUCCESS] Best Model Selected: {best_model_name} (F1: {best_f1:.4f}) -> Saved to models/best_wine_model.pkl")

if __name__ == "__main__":
    run_preprocessing()
    run_feature_engineering()
    run_validation()
    run_training_and_selection()
