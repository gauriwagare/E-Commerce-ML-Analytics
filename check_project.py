
from pathlib import Path
import json
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

print("\n" + "=" * 45)
print(" E-COMMERCE PROJECT HEALTH CHECK")
print("=" * 45)

# 1. Check important files
required_files = [
    "app.py",
    "best_purchase_model.pkl",
    "model_features.json",
]

print("\n1. FILE CHECK")
for filename in required_files:
    path = BASE_DIR / filename
    print(f"{filename}: {'FOUND' if path.exists() else 'MISSING'}")

# 2. Read expected model features
features_path = BASE_DIR / "model_features.json"
features = None

if features_path.exists():
    try:
        with open(features_path, "r", encoding="utf-8") as file:
            features = json.load(file)

        print("\n2. MODEL FEATURES")
        print(features)

    except Exception as error:
        print("Could not read model_features.json:", error)

# 3. Inspect saved model
model_path = BASE_DIR / "best_purchase_model.pkl"

if model_path.exists():
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        print("\n3. MODEL CHECK")
        print("Model type:", type(model).__name__)
        print("Classes:", getattr(model, "classes_", "Not available"))
        print(
            "Training feature names:",
            getattr(model, "feature_names_in_", "Not stored")
        )
        print(
            "Expected feature count:",
            getattr(model, "n_features_in_", "Not available")
        )

        if features is not None:
            expected_count = getattr(model, "n_features_in_", None)

            if expected_count is not None:
                print("Feature count matches:",
                      expected_count == len(features))

    except Exception as error:
        print("Could not load model:", repr(error))

# 4. Find and inspect CSV files
print("\n4. DATASET CHECK")

csv_files = list(BASE_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV files found in the project folder.")

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path, nrows=5)
        print(f"\nFile: {csv_path.name}")
        print("Columns:", df.columns.tolist())
        print("Sample rows:")
        print(df.head(2).to_string(index=False))
    except Exception as error:
        print(f"Could not inspect {csv_path.name}: {error}")

print("\n" + "=" * 45)
print("CHECK COMPLETE")
print("=" * 45)
