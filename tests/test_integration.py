import os
import json
import subprocess
def test_end_to_end_mlops_pipeline():
 result = subprocess.run(["python", "src/pipeline.py"], capture_output=True, text=True)
 assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
 assert os.path.exists("feature_store/registry.json"), "Feature store registry missing!"
 with open("feature_store/registry.json") as f:
  reg = json.load(f)
 assert os.path.exists(reg["storage_path"]), "Parquet feature file missing!"
 assert os.path.exists("mlruns"), "MLflow tracking directory not generated!"
print("Integration Test PASSED: All MLOps artifacts verified.")