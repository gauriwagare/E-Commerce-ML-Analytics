
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv(
    "clean_retail_data.csv",
    dtype={"InvoiceNo": str, "StockCode": str}
)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
df = df.dropna(subset=["CustomerID", "InvoiceDate"]).copy()
df["CustomerID"] = df["CustomerID"].astype(int)

df = df.sort_values("InvoiceDate")
cutoff = df["InvoiceDate"].max() - pd.Timedelta(days=30)

history = df[df["InvoiceDate"] <= cutoff]

customers = history.groupby("CustomerID").agg(
    TotalOrders=("InvoiceNo", "nunique"),
    TotalQuantity=("Quantity", "sum"),
    TotalSpend=("Revenue", "sum"),
    UniqueProducts=("StockCode", "nunique"),
    LastPurchase=("InvoiceDate", "max")
).reset_index()

customers["DaysSinceLastPurchase"] = (
    cutoff - customers["LastPurchase"]
).dt.days

future_customers = set(
    df.loc[df["InvoiceDate"] > cutoff, "CustomerID"].unique()
)

customers["PurchasedNext30Days"] = (
    customers["CustomerID"].isin(future_customers).astype(int)
)

with open("model_features.json", encoding="utf-8") as f:
    features = json.load(f)

model = joblib.load("best_purchase_model.pkl")
X = customers[features]
y = customers["PurchasedNext30Days"]

pred = model.predict(X)
prob = model.predict_proba(X)[:, list(model.classes_).index(1)]

print("WARNING: These are training-period metrics, not independent test metrics.")
print("Customers:", len(customers))
print("\nTarget distribution:")
print(y.value_counts())

print("\nAccuracy:", round(accuracy_score(y, pred), 4))
print("Precision:", round(precision_score(y, pred, zero_division=0), 4))
print("Recall:", round(recall_score(y, pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y, pred, zero_division=0), 4))

if y.nunique() == 2:
    print("ROC-AUC:", round(roc_auc_score(y, prob), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y, pred))

print("\nClassification report:")
print(classification_report(y, pred, zero_division=0))
