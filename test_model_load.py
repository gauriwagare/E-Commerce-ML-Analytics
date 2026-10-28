
from pathlib import Path
import pickle
import joblib

model_path = Path("best_purchase_model.pkl")

print("Model file size:", model_path.stat().st_size, "bytes")

# Test 1: pickle
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("\nPICKLE: SUCCESS")
    print("Model type:", type(model))
except Exception as e:
    print("\nPICKLE: FAILED")
    print(type(e).__name__, str(e))

# Test 2: joblib
try:
    model = joblib.load(model_path)

    print("\nJOBLIB: SUCCESS")
    print("Model type:", type(model))
    print("Classes:", getattr(model, "classes_", "Not available"))
except Exception as e:
    print("\nJOBLIB: FAILED")
    print(type(e).__name__, str(e))
