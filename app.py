"""
InventoryIQ-AI — Sales Intelligence Engine
XGBoost Pipeline • ColumnTransformer Inference Engine
"""

import pickle
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st

# pipeline_utils must be imported so pickle can resolve the FunctionTransformer
# stored inside the ColumnTransformer (used for date feature expansion).
import pipeline_utils  # noqa: F401

st.set_page_config(page_title="InventoryIQ-AI", page_icon="📈", layout="wide")

MODEL_PATH = Path(__file__).parent / "inventory_model.pkl"

FAMILIES = [
    "AUTOMOTIVE", "BEAUTY", "BEVERAGES", "BOOKS", "BREAD/BAKERY",
    "CLEANING", "DAIRY", "DELI", "EGGS", "FROZEN FOODS",
    "GROCERY I", "GROCERY II", "HOME CARE", "MEATS", "PET SUPPLIES",
    "PRODUCE", "SEAFOOD",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, f"Model file ({MODEL_PATH.name}) was not detected. Please ensure {MODEL_PATH.name} is located in the root directory."
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
else:
    st.success("✅ Model loaded successfully.")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎛️ Feature Configurations")

    transaction_id = st.number_input("Transaction ID", min_value=0, value=1001, step=1)
    selected_date = st.date_input("Select Date", value=date(2026, 8, 15))
    store_number = st.number_input("Store Number", min_value=1, max_value=54, value=1, step=1)
    on_promotion = st.number_input("On Promotion Count", min_value=0, value=0, step=1)
    product_family = st.selectbox("Product Family", FAMILIES, index=0)

    generate = st.button("✨ Generate Forecast", use_container_width=True, disabled=model is None)

with col_right:
    st.subheader("📊 Forecast Output")

    if model is None:
        st.error("Model is not loaded. Cannot execute inference.")
    elif generate:
        input_df = pd.DataFrame([{
            "transaction_id": transaction_id,
            "store_nbr": store_number,
            "onpromotion": on_promotion,
            "family": product_family,
            "date": pd.to_datetime(selected_date),
        }])
        try:
            prediction = model.predict(input_df)[0]
            st.metric("Forecasted Sales", f"{prediction:,.2f} units")
            with st.expander("Input features used"):
                st.dataframe(input_df, use_container_width=True)
        except Exception as e:
            st.error(f"Inference failed: {e}")
    else:
        st.info("Configure your inputs and click **Generate Forecast**.")

    st.write("")
    st.markdown(f"**Pipeline Architecture:** `ColumnTransformer` + `XGBRegressor`")
    st.markdown(f"**Categorical Handling:** `OneHotEncoder(handle_unknown='ignore')`")
