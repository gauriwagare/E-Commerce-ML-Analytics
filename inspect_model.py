
import joblib

model = joblib.load("best_purchase_model.pkl")

print("MODEL TYPE:", type(model))
print("\nPIPELINE STEPS:")

for name, step in model.named_steps.items():
    print(f"{name}: {type(step).__name__}")

# Inspect the final prediction algorithm
classifier = model.steps[-1][1]

print("\nCLASSIFIER:", type(classifier).__name__)

if hasattr(classifier, "get_params"):
    print("\nCLASSIFIER PARAMETERS:")
    for key, value in classifier.get_params().items():
        if key in [
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "class_weight",
            "random_state"
        ]:
            print(f"{key}: {value}")

if hasattr(classifier, "feature_importances_"):
    print("\nFEATURE IMPORTANCE:")
    for name, importance in zip(
        model_features := [
            "TotalOrders",
            "TotalQuantity",
            "TotalSpend",
            "UniqueProducts",
            "DaysSinceLastPurchase"
        ],
        classifier.feature_importances_
    ):
        print(f"{name}: {importance:.4f}")
