import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "inventory_model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file ({MODEL_PATH}) was not detected. Please ensure {MODEL_PATH} is located in the root directory."
        )
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model file '{MODEL_PATH}': {e}")
        return None


model = load_model()

if model is None:
    st.warning("Model is not loaded. Cannot execute inference.")
else:
    st.success("Model loaded successfully!")
    # Proceed with st.button() and model.predict(...)
