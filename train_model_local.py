
import json
import joblib
import pandas as pd
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

print("scikit-learn version:", sklearn.__version__)
print("Loading cleaned dataset...")

df = pd.read_csv(
    "clean_retail_data.csv",
    dtype={"InvoiceNo": str, "StockCode": str}
)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
df = df.dropna(subset=["CustomerID", "InvoiceDate"]).copy()
df["CustomerID"] = df["CustomerID"].astype(int)

# Create the same 30-day prediction target
df = df.sort_values("InvoiceDate").copy()
cutoff_date = df["InvoiceDate"].max() - pd.Timedelta(days=30)

history = df[df["InvoiceDate"] <= cutoff_date].copy()
future = df[df["InvoiceDate"] > cutoff_date].copy()

# Build historical customer features
customer_features = history.groupby("CustomerID").agg(
    TotalOrders=("InvoiceNo", "nunique"),
    TotalQuantity=("Quantity", "sum"),
    TotalSpend=("Revenue", "sum"),
    UniqueProducts=("StockCode", "nunique"),
    LastPurchase=("InvoiceDate", "max")
).reset_index()

customer_features["DaysSinceLastPurchase"] = (
    cutoff_date - customer_features["LastPurchase"]
).dt.days

customer_features = customer_features.drop(columns=["LastPurchase"])

future_customers = set(future["CustomerID"].unique())

customer_features["PurchasedNext30Days"] = (
    customer_features["CustomerID"]
    .isin(future_customers)
    .astype(int)
)

features = [
    "TotalOrders",
    "TotalQuantity",
    "TotalSpend",
    "UniqueProducts",
    "DaysSinceLastPurchase"
]

X = customer_features[features]
y = customer_features["PurchasedNext30Days"]

# Train a fresh model using the LOCAL scikit-learn version
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=6,
        random_state=42
    ))
])

model.fit(X, y)

# Save model and exact feature order
joblib.dump(model, "best_purchase_model.pkl")

with open("model_features.json", "w", encoding="utf-8") as f:
    json.dump(features, f)

print("\nModel retrained and saved successfully!")
print("Cutoff date:", cutoff_date)
print("Customers used:", len(customer_features))
print("Features:", features)
print("Target distribution:")
print(y.value_counts())
print("\nSaved best_purchase_model.pkl and model_features.json")
