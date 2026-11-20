import os
import pandas as pd
from sklearn.datasets import load_wine
def run_preprocessing():
 os.makedirs("data", exist_ok=True)
 wine = load_wine(as_frame=True)
 df = wine.frame.copy()
 df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]
 df = df.drop_duplicates().dropna().reset_index(drop=True)
 df.insert(0, "wine_id", range(1, len(df) + 1))
 df["event_timestamp"] = pd.Timestamp.utcnow()
 df.to_csv("data/wine_cleaned.csv", index=False)
 print(f"[Preprocessing] Saved data/wine_cleaned.csv | Shape: {df.shape}")
if __name__ == "__main__":
 run_preprocessing()
