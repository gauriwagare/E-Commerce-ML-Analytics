
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="E-commerce ML Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_purchase_model.pkl"
FEATURE_PATH = BASE_DIR / "model_features.json"
DATA_PATH = BASE_DIR / "clean_retail_data.csv"

REQUIRED_FEATURES = [
    "TotalOrders",
    "TotalQuantity",
    "TotalSpend",
    "UniqueProducts",
    "DaysSinceLastPurchase",
]


# =========================================================
# 2. CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f9fc, #eef3fa);
    }

    section[data-testid="stSidebar"] {
        background-color: #14213d;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .hero {
        padding: 25px;
        border-radius: 16px;
        background: linear-gradient(120deg, #14213d, #274c77);
        color: white;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
    }

    .hero p {
        color: #e8eef7;
        margin-top: 8px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e3e9f2;
        border-radius: 14px;
        padding: 18px;
        min-height: 105px;
        box-shadow: 0 3px 12px rgba(20, 33, 61, 0.05);
    }

    .metric-label {
        color: #53627a;
        font-size: 0.85rem;
    }

    .metric-value {
        color: #14213d;
        font-size: 1.7rem;
        font-weight: 750;
        margin-top: 8px;
    }

    .small-note {
        color: #7b8798;
        font-size: 0.75rem;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD MODEL AND DATA
# =========================================================
@st.cache_resource
def load_model(path):
    path = Path(path)

    if not path.exists():
        return None

    # Model was saved using joblib.
    return joblib.load(path)


@st.cache_data
def load_data(path):
    path = Path(path)

    if not path.exists():
        return None

    data = pd.read_csv(path)

    if "InvoiceDate" in data.columns:
        data["InvoiceDate"] = pd.to_datetime(
            data["InvoiceDate"], errors="coerce"
        )

    if "Revenue" not in data.columns:
        if {"Quantity", "UnitPrice"}.issubset(data.columns):
            data["Revenue"] = data["Quantity"] * data["UnitPrice"]

    return data


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Load the dataset
try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Dataset loading error: {exc}")
    st.stop()

if df is None:
    st.error(
        f"File not found: {DATA_PATH.name}. "
        "Keep the CSV file in the same folder as app.py."
    )
    st.stop()


# Check required columns
required_columns = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "Revenue",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(
        "Dataset is missing these columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# Clean relevant data types
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

for col in ["Quantity", "UnitPrice", "Revenue"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "InvoiceNo",
        "StockCode",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "Revenue",
    ]
)

# Positive sales only for product recommendations and pricing simulation.
sales_df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()

if sales_df.empty:
    st.error("No valid positive-quantity sales records were found.")
    st.stop()


# Load the model
try:
    model = load_model(MODEL_PATH)
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)


# Load model feature order
try:
    with open(FEATURE_PATH, "r", encoding="utf-8") as file:
        model_features = json.load(file)
except FileNotFoundError:
    model_features = REQUIRED_FEATURES
except Exception as exc:
    model_features = None
    st.warning(f"Could not read model_features.json: {exc}")


# Product-level summary used by the dashboard
product_stats = (
    sales_df.groupby("StockCode")
    .agg(
        Description=("Description", "first"),
        AveragePrice=("UnitPrice", "mean"),
        UnitsSold=("Quantity", "sum"),
        InvoiceCount=("InvoiceNo", "nunique"),
        TotalRevenue=("Revenue", "sum"),
    )
    .reset_index()
)

product_stats["Description"] = (
    product_stats["Description"].fillna("Unknown product")
)

product_stats["ProductLabel"] = (
    product_stats["Description"].astype(str)
    + " ("
    + product_stats["StockCode"].astype(str)
    + ")"
)

product_labels = dict(
    zip(product_stats["StockCode"], product_stats["ProductLabel"])
)


# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("# 🛍️ E-commerce AI")
    st.caption("ML Analytics Dashboard")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Purchase Prediction",
            "Product Recommendations",
            "Discount & Profit Simulator",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Online Retail transaction dataset")
    st.caption("Machine Learning Lab Project")


# =========================================================
# 5. DASHBOARD HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>🛍️ E-commerce ML Analytics</h1>
        <p>
            Understand customers, discover products,
            and simulate pricing decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 6. OVERVIEW PAGE
# =========================================================
if page == "Overview":

    st.subheader("📊 Project Overview")

    st.write(
        "A dashboard for customer analytics, purchase prediction, "
        "product recommendations, and pricing scenario analysis."
    )

    total_transactions = len(df)

    customers = (
        df["CustomerID"].nunique()
        if "CustomerID" in df.columns
        else 0
    )

    products = sales_df["StockCode"].nunique()
    invoices = sales_df["InvoiceNo"].nunique()
    total_revenue = sales_df["Revenue"].sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Clean Transactions",
            f"{total_transactions:,}",
        )

    with c2:
        metric_card(
            "Unique Customers",
            f"{customers:,}",
        )

    with c3:
        metric_card(
            "Unique Products",
            f"{products:,}",
        )

    with c4:
        metric_card(
            "Unique Invoices",
            f"{invoices:,}",
        )

    st.metric(
        "Recorded Sales Revenue",
        f"{total_revenue:,.2f}",
    )

    st.caption(
        "Revenue shown here uses positive-quantity sales records. "
        "It describes historical transactions, not future revenue."
    )

    st.divider()

    left, right = st.columns(2)

    # Monthly revenue chart
    with left:
        st.subheader("📈 Revenue by Month")

        monthly = (
            sales_df.set_index("InvoiceDate")["Revenue"]
            .resample("MS")
            .sum()
            .reset_index()
        )

        fig = px.line(
            monthly,
            x="InvoiceDate",
            y="Revenue",
            markers=True,
            template="plotly_white",
            labels={
                "InvoiceDate": "Month",
                "Revenue": "Revenue",
            },
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=25, b=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # Top products chart
    with right:
        st.subheader("🏆 Top 10 Products")

        top_products = (
            sales_df.groupby("Description")["InvoiceNo"]
            .nunique()
            .nlargest(10)
            .sort_values()
            .reset_index(name="InvoiceCount")
        )

        fig = px.bar(
            top_products,
            x="InvoiceCount",
            y="Description",
            orientation="h",
            template="plotly_white",
            labels={
                "InvoiceCount": "Unique Invoices",
                "Description": "Product",
            },
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=25, b=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # Model comparison
    st.subheader("🤖 Machine Learning Model Comparison")

    comparison = pd.DataFrame(
        [
            [
                "Logistic Regression",
                0.6931, 0.5408, 0.6297, 0.5819, 0.7418
            ],
            [
                "Decision Tree",
                0.6742, 0.5151, 0.6674, 0.5814, 0.7312
            ],
            [
                "Random Forest",
                0.6875, 0.5335, 0.6232, 0.5749, 0.7350
            ],
        ],
        columns=[
            "Algorithm",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
        ],
    )

    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
                "ROC-AUC": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        comparison,
        x="Algorithm",
        y="F1 Score",
        text=comparison["F1 Score"].map(
            lambda value: f"{value:.3f}"
        ),
        title="Purchase-Class F1 Score Comparison",
        template="plotly_white",
    )

    fig.update_layout(
        yaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=45, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        "These are the recorded holdout-period results from "
        "compare_models.py. Logistic Regression has the highest "
        "ROC-AUC and a marginally higher F1 score. Decision Tree "
        "has the highest recall. The prediction page uses the model "
        "saved in best_purchase_model.pkl, currently a Decision Tree."
    )


# =========================================================
# 7. PURCHASE PREDICTION PAGE
# =========================================================
elif page == "Purchase Prediction":

    st.subheader("🎯 Next 30-Day Purchase Prediction")

    st.write(
        "Enter a customer's historical activity to estimate "
        "whether they may purchase during the next 30 days."
    )

    if model is None:
        st.error(
            "The prediction model could not be loaded. "
            "Check best_purchase_model.pkl."
        )
        if model_error:
            st.code(model_error)
        st.stop()

    if list(model_features or []) != REQUIRED_FEATURES:
        st.error(
            "Feature names/order in model_features.json do not "
            "match the features expected by this app."
        )
        st.write("Expected order:", REQUIRED_FEATURES)
        st.write("Loaded order:", model_features)
        st.stop()

    st.caption(
        "Features: order count, quantity, spending, product variety, "
        "and days since the last purchase."
    )

    col1, col2 = st.columns(2)

    with col1:
        total_orders = st.number_input(
            "Total Orders",
            min_value=0,
            max_value=10000,
            value=5,
            step=1,
        )

        total_quantity = st.number_input(
            "Total Quantity Purchased",
            min_value=0,
            max_value=1000000,
            value=30,
            step=1,
        )

        total_spend = st.number_input(
            "Total Spend",
            min_value=0.0,
            max_value=100000000.0,
            value=250.0,
            step=10.0,
        )

    with col2:
        unique_products = st.number_input(
            "Unique Products Purchased",
            min_value=0,
            max_value=100000,
            value=8,
            step=1,
        )

        days_since_last_purchase = st.number_input(
            "Days Since Last Purchase",
            min_value=0,
            max_value=5000,
            value=20,
            step=1,
        )

    if st.button(
        "🔍 Predict Purchase Probability",
        type="primary",
        use_container_width=True,
    ):

        input_df = pd.DataFrame(
            [[
                total_orders,
                total_quantity,
                total_spend,
                unique_products,
                days_since_last_purchase,
            ]],
            columns=REQUIRED_FEATURES,
        )

        try:
            prediction = int(model.predict(input_df)[0])

            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]

                if hasattr(model, "classes_"):
                    classes = list(model.classes_)

                    if 1 in classes:
                        positive_index = classes.index(1)
                        probability = float(
                            probabilities[positive_index]
                        )

            if prediction == 1:
                st.success(
                    "Prediction: Customer is likely to purchase "
                    "in the next 30 days according to the model."
                )
            else:
                st.warning(
                    "Prediction: Customer is not predicted to "
                    "purchase in the next 30 days."
                )

            if probability is not None:
                p1, p2 = st.columns(2)

                with p1:
                    st.metric(
                        "Estimated Purchase Probability",
                        f"{probability:.1%}",
                    )

                with p2:
                    st.metric(
                        "Estimated Non-Purchase Probability",
                        f"{1 - probability:.1%}",
                    )

                st.progress(
                    max(0.0, min(1.0, probability))
                )

            st.subheader("Customer Input Details")

            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "This is a model estimate, not a guarantee. "
                "Decision Tree probabilities can be extreme and "
                "may not be well calibrated."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# =========================================================
# 8. PRODUCT RECOMMENDATIONS PAGE
# =========================================================
elif page == "Product Recommendations":

    st.subheader("🧺 Product Recommendations")

    st.write(
        "Discover products that frequently appear in the same "
        "invoice as the selected product."
    )

    st.caption(
        "Method: invoice-based co-purchase frequency."
    )

    invoice_products = (
        sales_df[
            ["InvoiceNo", "StockCode", "Description"]
        ]
        .dropna(subset=["InvoiceNo", "StockCode"])
        .drop_duplicates(["InvoiceNo", "StockCode"])
    )

    product_options = product_stats["StockCode"].tolist()

    if not product_options:
        st.warning("No products are available.")
        st.stop()

    selected_code = st.selectbox(
        "Choose a Product",
        options=product_options,
        format_func=lambda code: product_labels.get(
            code, str(code)
        ),
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=15,
        value=5,
    )

    # Find invoices containing the selected product.
    target_invoices = set(
        invoice_products.loc[
            invoice_products["StockCode"] == selected_code,
            "InvoiceNo",
        ]
    )

    # Count other products occurring in those invoices.
    co_counts = (
        invoice_products[
            invoice_products["InvoiceNo"].isin(target_invoices)
        ]
        .loc[
            lambda data: data["StockCode"] != selected_code
        ]
        .groupby(["StockCode", "Description"])["InvoiceNo"]
        .nunique()
        .reset_index(name="SharedInvoiceCount")
        .sort_values(
            "SharedInvoiceCount",
            ascending=False,
        )
    )

    if co_counts.empty:
        st.info(
            "No co-purchased products were found for this selection."
        )

    else:
        total_target_invoices = max(
            len(target_invoices), 1
        )

        co_counts["ShareOfSelectedProductInvoices"] = (
            co_counts["SharedInvoiceCount"]
            / total_target_invoices
        )

        co_counts["Product"] = (
            co_counts["Description"]
            .fillna("Unknown product")
            .astype(str)
        )

        shown = co_counts.head(top_n).copy()

        shown["Co-purchase Share"] = (
            shown["ShareOfSelectedProductInvoices"]
            .map(lambda value: f"{value:.1%}")
        )

        st.dataframe(
            shown[
                [
                    "Product",
                    "SharedInvoiceCount",
                    "Co-purchase Share",
                ]
            ].rename(
                columns={
                    "SharedInvoiceCount": "Invoices Bought Together"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            shown.sort_values("SharedInvoiceCount"),
            x="SharedInvoiceCount",
            y="Product",
            orientation="h",
            title="Frequently Co-purchased Products",
            template="plotly_white",
            labels={
                "SharedInvoiceCount": "Invoices Bought Together",
                "Product": "Product",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.info(
        "These recommendations are based on historical invoice "
        "co-occurrence. Association does not prove causation."
    )


# =========================================================
# 9. DISCOUNT & PROFIT SIMULATOR
# =========================================================
elif page == "Discount & Profit Simulator":

    st.subheader("💸 Discount & Profit Simulator")

    st.write(
        "Compare a product's baseline revenue and profit with "
        "a discounted scenario."
    )

    st.caption(
        "The expected sales increase and product cost are assumptions "
        "entered by the user, not values learned from the dataset."
    )

    product_options = product_stats.sort_values("Description")

    options = product_options["StockCode"].tolist()

    if not options:
        st.warning("No products are available.")
        st.stop()

    selected_code = st.selectbox(
        "Choose a Product",
        options=options,
        format_func=lambda code: product_labels.get(
            code, str(code)
        ),
        key="sim_product",
    )

    item = product_stats[
        product_stats["StockCode"] == selected_code
    ].iloc[0]

    avg_price = float(item["AveragePrice"])
    units = float(item["UnitsSold"])

    st.caption(
        f"Historical average selling price: {avg_price:.2f} "
        "dataset currency units."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        unit_cost = st.number_input(
            "Assumed Cost per Unit",
            min_value=0.0,
            value=round(avg_price * 0.40, 2),
            step=0.1,
            key="unit_cost",
        )

    with c2:
        discount = st.slider(
            "Discount (%)",
            min_value=0,
            max_value=70,
            value=10,
        )

    with c3:
        demand_increase = st.slider(
            "Assumed Unit-Sales Increase (%)",
            min_value=0,
            max_value=100,
            value=15,
        )

    # Calculate the baseline scenario.
    sale_price = avg_price * (1 - discount / 100)

    simulated_units = (
        units * (1 + demand_increase / 100)
    )

    original_revenue = avg_price * units
    discounted_revenue = sale_price * simulated_units

    original_profit = (
        avg_price - unit_cost
    ) * units

    discounted_profit = (
        sale_price - unit_cost
    ) * simulated_units

    profit_difference = (
        discounted_profit - original_profit
    )

    result = pd.DataFrame(
        {
            "Metric": [
                "Average Selling Price",
                "Discounted Selling Price",
                "Baseline Units Sold",
                "Simulated Units Sold",
                "Revenue Without Discount",
                "Revenue With Discount",
                "Profit Without Discount",
                "Profit With Discount",
                "Profit Difference",
            ],
            "Value": [
                avg_price,
                sale_price,
                units,
                simulated_units,
                original_revenue,
                discounted_revenue,
                original_profit,
                discounted_profit,
                profit_difference,
            ],
        }
    )

    if st.button(
        "🧮 Run Simulation",
        type="primary",
        use_container_width=True,
    ):

        st.subheader("Simulation Results")

        st.dataframe(
            result.style.format(
                {"Value": "{:,.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_data = pd.DataFrame(
            {
                "Metric": [
                    "Revenue Without Discount",
                    "Revenue With Discount",
                    "Profit Without Discount",
                    "Profit With Discount",
                ],
                "Value": [
                    original_revenue,
                    discounted_revenue,
                    original_profit,
                    discounted_profit,
                ],
            }
        )

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Value",
            color="Metric",
            title="Revenue and Profit Comparison",
            template="plotly_white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        if profit_difference > 0:
            st.success(
                f"Under these assumptions, profit increases by "
                f"{profit_difference:,.2f} dataset currency units."
            )

        elif profit_difference < 0:
            st.warning(
                f"Under these assumptions, profit decreases by "
                f"{abs(profit_difference):,.2f} dataset currency units."
            )

        else:
            st.info(
                "Under these assumptions, profit remains unchanged."
            )

        st.info(
            "Simulation limitation: the dataset does not contain "
            "actual product costs or causal discount-response data. "
            "Results depend on the cost and demand-increase assumptions."
        )


# =========================================================
# 10. FOOTER
# =========================================================
st.divider()

st.caption(
    "E-commerce ML Analytics Dashboard | Educational ML project | "
    "Historical sales data, ML estimates, co-purchase recommendations, "
    "and simulated pricing scenarios"
)
