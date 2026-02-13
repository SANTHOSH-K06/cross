import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer

# Page configuration
st.set_page_config(
    page_title="Wine Quality Cross-Validation",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .metric-container {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #6c5ce7;
        color: white;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #a29bfe;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

st.title("🍷 Wine Quality Predictor")
st.markdown("### Predict wine quality based on physico-chemical tests")

# Dataset Loading and Model Training
@st.cache_resource
def train_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    X = data.drop('quality', axis=1)
    y = data['quality']
    model = LinearRegression()
    model.fit(X, y)
    return model, data.describe()

try:
    with st.spinner("Preparing prediction engine..."):
        model, stats = train_model()
    
    st.sidebar.header("🧪 Wine Characteristics")
    st.sidebar.markdown("Adjust the parameters to predict quality")

    # Dynamic Input Form based on dataset stats
    inputs = {}
    
    # Organize inputs into columns for a better UI
    col1, col2 = st.columns(2)
    
    features = [
        ("fixed acidity", "Fixed Acidity"),
        ("volatile acidity", "Volatile Acidity"),
        ("citric acid", "Citric Acid"),
        ("residual sugar", "Residual Sugar"),
        ("chlorides", "Chlorides"),
        ("free sulfur dioxide", "Free Sulfur Dioxide"),
        ("total sulfur dioxide", "Total Sulfur Dioxide"),
        ("density", "Density"),
        ("pH", "pH"),
        ("sulphates", "Sulphates"),
        ("alcohol", "Alcohol")
    ]

    for i, (col_name, label) in enumerate(features):
        min_val = float(stats.loc['min', col_name])
        max_val = float(stats.loc['max', col_name])
        mean_val = float(stats.loc['mean', col_name])
        
        # Determine step size based on range
        val_range = max_val - min_val
        step = 0.01 if val_range < 10 else 1.0
        
        with col1 if i % 2 == 0 else col2:
            inputs[col_name] = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=step,
                help=f"Range: {min_val} - {max_val}"
            )

    st.divider()
    
    # Prediction logic
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    
    # Premium Result Display
    st.subheader("🎯 Prediction Result")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric("Predicted Quality Score", f"{prediction:.2f}")
        
    with res_col2:
        # Visual feedback based on score
        if prediction >= 7:
            st.success("✨ **Premium Quality!** This wine shows exceptional characteristics.")
        elif prediction >= 5:
            st.info("👍 **Standard Quality.** This is a well-balanced wine.")
        else:
            st.warning("⚠️ **Lower Quality.** The chemical profile suggests some inconsistencies.")
            
        st.progress(min(max(prediction / 10, 0.0), 1.0))

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please check the dataset connectivity.")


