
import json
import joblib
import pandas as pd

# Load model and feature names
model = joblib.load("best_purchase_model.pkl")

with open("model_features.json", "r") as f:
    feature_names = json.load(f)

# Test customer data
customer = {
    "TotalOrders": 18,
    "TotalQuantity": 121,
    "TotalSpend": 260,
    "UniqueProducts": 13,
    "DaysSinceLastPurchase": 16
}

input_df = pd.DataFrame([customer])[feature_names]

print("Feature order:", feature_names)
print("\nInput:")
print(input_df)

prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]

print("\nPrediction:", prediction)
print("Classes:", model.classes_)
print("Class probabilities:", probabilities)

purchase_index = list(model.classes_).index(1)
purchase_probability = probabilities[purchase_index] * 100

print(f"Purchase probability: {purchase_probability:.2f}%")
print(
    "Result:",
    "Likely to purchase" if prediction == 1
    else "Not likely to purchase"
)
