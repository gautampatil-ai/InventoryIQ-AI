import streamlit as st
import pandas as pd
import pickle
import os
from datetime import date

st.set_page_config(page_title="InventoryIQ-AI", page_icon="📈", layout="wide")

MODEL_PATH = "inventory_model.pkl"

PRODUCT_FAMILIES = [
    "AUTOMOTIVE", "BEAUTY", "BEVERAGES", "BOOKS", "BREAD/BAKERY",
    "CLEANING", "DAIRY", "DELI", "EGGS", "FROZEN FOODS",
    "GROCERY I", "GROCERY II", "HARDWARE", "HOME CARE",
    "LADIESWEAR", "LIQUOR,WINE,BEER", "MEATS", "PERSONAL CARE",
    "PRODUCE", "SEAFOOD",
]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file ({MODEL_PATH}) was not detected. Please ensure {MODEL_PATH} is located in the root directory."
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"Error loading model file '{MODEL_PATH}': {e}"


model, load_error = load_model()

st.title("📈 Sales Intelligence Engine")
st.caption("XGBoost Pipeline • ColumnTransformer Inference Engine")

if load_error:
    st.error(f"❌ {load_error}")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎛️ Feature Configurations")

    transaction_id = st.number_input("Transaction ID", min_value=1, value=1001, step=1)
    select_date = st.date_input("Select Date", value=date(2026, 8, 15))
    store_number = st.number_input("Store Number", min_value=1, max_value=54, value=1, step=1)
    on_promotion = st.number_input("On Promotion Count", min_value=0, value=0, step=1)
    product_family = st.selectbox("Product Family", PRODUCT_FAMILIES, index=0)

    generate = st.button("✨ Generate Forecast", use_container_width=True, disabled=model is None)

with col_right:
    st.subheader("📊 Forecast Output")

    if model is None:
        st.error("Model is not loaded. Cannot execute inference.")
    elif generate:
        day_of_week = select_date.weekday()
        month = select_date.month
        day = select_date.day
        is_weekend = int(day_of_week >= 5)

        X = pd.DataFrame([{
            "store_nbr": store_number,
            "family": product_family,
            "onpromotion": on_promotion,
            "day_of_week": day_of_week,
            "month": month,
            "day": day,
            "is_weekend": is_weekend,
        }])

        prediction = model.predict(X)[0]
        st.metric("Forecasted Sales", f"{prediction:,.2f} units")
        st.dataframe(X, use_container_width=True)
    else:
        st.info("Configure the inputs on the left and click **Generate Forecast**.")

    st.divider()
    st.markdown("**Pipeline Architecture:** `ColumnTransformer` + `XGBRegressor`")
    st.markdown("**Categorical Handling:** `OneHotEncoder(handle_unknown='ignore')`")
