import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
def run_feature_engineering():
 os.makedirs("feature_store", exist_ok=True)
 df = pd.read_csv("data/wine_cleaned.csv")
 df["alcohol_to_malic_ratio"] = df["alcohol"] / (df["malic_acid"] + 1e-5)
 df["total_phenols_flavonoids"] = df["total_phenols"] + df["flavanoids"]
 df["color_hue_interaction"] = df["color_intensity"] * df["hue"]
 meta_cols = ["wine_id", "event_timestamp", "target"]
 feature_cols = [c for c in df.columns if c not in meta_cols]
 scaler = StandardScaler()
 df[feature_cols] = scaler.fit_transform(df[feature_cols])
 joblib.dump(scaler, "feature_store/wine_scaler.pkl")
 store_path = "feature_store/wine_features.parquet"
 df.to_parquet(store_path, index=False)
 registry = {
 "feature_group": "wine_classification_features",
 "version": "v1.0",
 "entity_key": "wine_id",
 "timestamp_key": "event_timestamp",
 "target": "target",
 "features": feature_cols,
 "storage_path": store_path
 }
 with open("feature_store/registry.json", "w") as f:
  json.dump(registry, f, indent=4)
 print(f"[Feature Store] Materialized {len(feature_cols)} features to {store_path}")
if __name__ == "__main__":
 run_feature_engineering()
