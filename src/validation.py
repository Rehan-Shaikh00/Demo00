import json
import pandas as pd
def run_validation():
 with open("feature_store/registry.json", "r") as f:
   registry = json.load(f)
 df = pd.read_parquet(registry["storage_path"])
 assert df.isnull().sum().sum() == 0, "Null values detected!"
 expected = set(registry["features"] + [registry["entity_key"], registry["target"]])
 assert expected.issubset(set(df.columns)), "Feature Store schema mismatch!"
 assert df["target"].nunique() == 3, "Expected 3 wine classes (0, 1, 2)!"
 print("[Validation] All Wine schema and quality checks PASSED.")
if __name__ == "__main__":
 run_validation()