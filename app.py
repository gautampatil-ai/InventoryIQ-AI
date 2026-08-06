import streamlit as st
import joblib
import pandas as pd
import datetime

# ---------------------------------------------------------------------------
# Page Configuration & UI Customization
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Sales Forecasting Engine",
    page_icon="📈",
    layout="wide"import streamlit as st
import joblib
import pandas as pd
import datetime
import os

# ---------------------------------------------------------------------------
# Page Configuration & UI Customization
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Sales Forecasting Engine",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp {
            background-color: #0f172a;
            color: #f8fafc;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px dashed #334155;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }
        .prediction-value {
            font-size: 3rem;
            font-weight: 700;
            color: #06b6d4;
            text-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model & Asset Loading (Updated to match repository filename)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "inventory_model.pkl")

@st.cache_resource
def load_pipeline():
    try:
        model = joblib.load(MODEL_PATH)
        return model, True
    except Exception as e:
        return None, False

model, model_loaded = load_pipeline()

# Feature mappings
CAT_FEATURES = ['family']
NUM_FEATURES = ['id', 'store_nbr', 'onpromotion', 'year', 'month', 'day', 'week', 'dayofweek']
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

FAMILY_OPTIONS = [
    'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 
    'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 
    'HARDWARE', 'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES', 
    'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 
    'MAGAZINES', 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'POULTRY', 'PREPARED FOODS', 
    'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'
]

# ---------------------------------------------------------------------------
# Layout Definition
# ---------------------------------------------------------------------------
st.title("📈 Sales Intelligence Engine")
st.caption("XGBoost Pipeline • ColumnTransformer Inference Engine")

if not model_loaded:
    st.warning("⚠️ Model file (`inventory_model.pkl`) was not detected. Please ensure `inventory_model.pkl` is located in the root directory.")

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("🎛️ Feature Configurations")
    
    with st.form("prediction_form"):
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            trans_id = st.number_input("Transaction ID", value=1001, step=1)
            store_nbr = st.number_input("Store Number", value=1, step=1)
            onpromotion = st.number_input("On Promotion Count", value=0, step=1)
        
        with form_col2:
            selected_date = st.date_input("Select Date", value=datetime.date(2026, 8, 15))
            year = selected_date.year
            month = selected_date.month
            day = selected_date.day
            week = selected_date.isocalendar()[1]
            dayofweek = selected_date.weekday()

        family = st.selectbox("Product Family", options=FAMILY_OPTIONS)

        submit_button = st.form_submit_button("✨ Generate Forecast", use_container_width=True)

with col2:
    st.subheader("📊 Forecast Output")
    
    if submit_button:
        if not model_loaded:
            st.error("Model is not loaded. Cannot execute inference.")
        else:
            try:
                input_data = {
                    'id': [int(trans_id)],
                    'store_nbr': [int(store_nbr)],
                    'family': [family],
                    'onpromotion': [int(onpromotion)],
                    'year': [int(year)],
                    'month': [int(month)],
                    'day': [int(day)],
                    'week': [int(week)],
                    'dayofweek': [int(dayofweek)]
                }
                
                df = pd.DataFrame(input_data)[ALL_FEATURES]
                prediction = model.predict(df)[0]
                
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="text-transform: uppercase; font-size: 0.85rem; color: #94a3b8; font-weight: 600;">Estimated Unit Sales</div>
                        <div class="prediction-value">{prediction:.2f}</div>
                        <div style="color: #22c55e; font-size: 0.85rem; margin-top: 0.5rem;">✔️ Inference Completed</div>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Inference error: {str(e)}")
    else:
        st.markdown("""
            <div class="metric-card">
                <div style="text-transform: uppercase; font-size: 0.85rem; color: #94a3b8; font-weight: 600;">Estimated Unit Sales</div>
                <div class="prediction-value" style="color: #64748b;">--</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">Submit parameters to trigger inference</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
        **Pipeline Architecture:** `ColumnTransformer` + `XGBRegressor`  
        **Categorical Handling:** `OneHotEncoder(handle_unknown='ignore')`
    """)
)

# Optional: Add custom styling for a dark high-tech dashboard look
st.markdown("""
    <style>
        .stApp {
            background-color: #0f172a;
            color: #f8fafc;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px dashed #334155;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }
        .prediction-value {
            font-size: 3rem;
            font-weight: 700;
            color: #06b6d4;
            text-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model & Asset Loading
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_pipeline():
    try:
        model = joblib.load(MODEL_PATH)
        return model, True
    except Exception as e:
        return None, False

model, model_loaded = load_pipeline()

# Feature mappings
CAT_FEATURES = ['family']
NUM_FEATURES = ['id', 'store_nbr', 'onpromotion', 'year', 'month', 'day', 'week', 'dayofweek']
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

FAMILY_OPTIONS = [
    'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 
    'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 
    'HARDWARE', 'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES', 
    'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 
    'MAGAZINES', 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'POULTRY', 'PREPARED FOODS', 
    'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'
]

# ---------------------------------------------------------------------------
# Layout Definition
# ---------------------------------------------------------------------------
st.title("📈 Sales Intelligence Engine")
st.caption("XGBoost Pipeline • ColumnTransformer Inference Engine")

if not model_loaded:
    st.warning("⚠️ Model file (`model.pkl`) was not detected. Please ensure `model.pkl` is located in the root directory.")

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("🎛️ Feature Configurations")
    
    with st.form("prediction_form"):
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            trans_id = st.number_input("Transaction ID", value=1001, step=1)
            store_nbr = st.number_input("Store Number", value=1, step=1)
            onpromotion = st.number_input("On Promotion Count", value=0, step=1)
        
        with form_col2:
            selected_date = st.date_input("Select Date", value=datetime.date(2026, 8, 15))
            # Automatically compute calendar components from chosen date
            year = selected_date.year
            month = selected_date.month
            day = selected_date.day
            week = selected_date.isocalendar()[1]
            dayofweek = selected_date.weekday()

        family = st.selectbox("Product Family", options=FAMILY_OPTIONS)

        submit_button = st.form_submit_button("✨ Generate Forecast", use_container_width=True)

with col2:
    st.subheader("📊 Forecast Output")
    
    if submit_button:
        if not model_loaded:
            st.error("Model is not loaded. Cannot execute inference.")
        else:
            try:
                # Structure inputs to match expected pandas DataFrame
                input_data = {
                    'id': [int(trans_id)],
                    'store_nbr': [int(store_nbr)],
                    'family': [family],
                    'onpromotion': [int(onpromotion)],
                    'year': [int(year)],
                    'month': [int(month)],
                    'day': [int(day)],
                    'week': [int(week)],
                    'dayofweek': [int(dayofweek)]
                }
                
                df = pd.DataFrame(input_data)[ALL_FEATURES]
                prediction = model.predict(df)[0]
                
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="text-transform: uppercase; font-size: 0.85rem; color: #94a3b8; font-weight: 600;">Estimated Unit Sales</div>
                        <div class="prediction-value">{prediction:.2f}</div>
                        <div style="color: #22c55e; font-size: 0.85rem; margin-top: 0.5rem;">✔️ Inference Completed</div>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Inference error: {str(e)}")
    else:
        st.markdown("""
            <div class="metric-card">
                <div style="text-transform: uppercase; font-size: 0.85rem; color: #94a3b8; font-weight: 600;">Estimated Unit Sales</div>
                <div class="prediction-value" style="color: #64748b;">--</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">Submit parameters to trigger inference</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
        **Pipeline Architecture:** `ColumnTransformer` + `XGBRegressor`  
        **Categorical Handling:** `OneHotEncoder(handle_unknown='ignore')`
    """)
