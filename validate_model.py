
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# 1. Load data
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

print("Dataset ends:", last_date)
print("Test history cutoff:", test_cutoff)
print("Test target period:", test_cutoff, "to", last_date)


# 2. Build customer features and next-30-day target
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

    future_buyers = set(future["CustomerID"].unique())
    customers["Target"] = (
        customers["CustomerID"].isin(future_buyers).astype(int)
    )

    return customers


# 3. Create historical training snapshots.
# Training target windows end no later than test_cutoff.
first_date = df["InvoiceDate"].min().normalize()
first_cutoff = first_date + pd.Timedelta(days=60)

training_cutoffs = []
cutoff = first_cutoff

while cutoff + pd.Timedelta(days=30) <= test_cutoff:
    training_cutoffs.append(cutoff)
    cutoff += pd.Timedelta(days=30)

if len(training_cutoffs) < 2:
    raise ValueError("Not enough historical snapshots for validation.")

train_parts = [make_snapshot(c) for c in training_cutoffs]
train_data = pd.concat(train_parts, ignore_index=True)

# 4. Latest unseen 30-day period
test_data = make_snapshot(test_cutoff)

X_train = train_data[features]
y_train = train_data["Target"]
X_test = test_data[features]
y_test = test_data["Target"]

print("\nTraining snapshots:", len(training_cutoffs))
print("Training customer-snapshots:", len(train_data))
print("Test customers:", len(test_data))
print("\nTraining target distribution:")
print(y_train.value_counts())
print("\nTest target distribution:")
print(y_test.value_counts())

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError("Training or test data contains only one target class.")

# 5. Train a fresh Decision Tree on historical snapshots
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=6,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 6. Evaluate on the held-out period
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[
    :, list(model.classes_).index(1)
]

print("\n===== TIME-BASED TEST RESULTS =====")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))
