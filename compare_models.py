
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
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

features = [
    "TotalOrders", "TotalQuantity", "TotalSpend",
    "UniqueProducts", "DaysSinceLastPurchase"
]

last_date = df["InvoiceDate"].max()
test_cutoff = last_date - pd.Timedelta(days=30)


def make_snapshot(cutoff):
    history = df[df["InvoiceDate"] <= cutoff]
    future = df[
        (df["InvoiceDate"] > cutoff) &
        (df["InvoiceDate"] <= cutoff + pd.Timedelta(days=30))
    ]

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

    buyers = set(future["CustomerID"].unique())
    customers["Target"] = customers["CustomerID"].isin(buyers).astype(int)

    return customers


# Build historical training snapshots
first_cutoff = df["InvoiceDate"].min().normalize() + pd.Timedelta(days=60)
cutoffs = []
cutoff = first_cutoff

while cutoff + pd.Timedelta(days=30) <= test_cutoff:
    cutoffs.append(cutoff)
    cutoff += pd.Timedelta(days=30)

train = pd.concat([make_snapshot(c) for c in cutoffs], ignore_index=True)
test = make_snapshot(test_cutoff)

X_train, y_train = train[features], train["Target"]
X_test, y_test = test[features], test["Target"]

models = {
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            class_weight="balanced", max_depth=6, random_state=42
        ))
    ]),
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=42,
            n_jobs=-1
        ))
    ])
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[
        :, list(model.classes_).index(1)
    ]

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "F1-score": round(f1_score(y_test, pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, prob), 4)
    })

comparison = pd.DataFrame(results).sort_values("F1-score", ascending=False)

print("TEST PERIOD:", test_cutoff, "to", last_date)
print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))
