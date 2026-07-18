import streamlit as pd
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Target Marketing Predictor",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for an attractive, modern UI
st.markdown("""
    <style>
    /* Main background and font */
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Card Styling */
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-container h1 {
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    /* Input Form Container */
    .stForm {
        background-color: white !important;
        padding: 30px !important;
        border-radius: 15px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #e5e7eb !important;
    }
    
    /* Custom Prediction Alert Boxes */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .positive-result {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .negative-result {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained Naive Bayes model safely
@st.cache_resource
def load_model():
    try:
        with open("naive_model_pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'naive_model_pkl' not found. Please ensure it's in the same directory.")
        return None

model = load_model()

# Header Display
st.markdown("""
    <div class="header-container">
        <h1>🎯 Customer Purchase Predictor</h1>
        <p>Enter customer demographics to predict likelihood of purchase engagement.</p>
    </div>
""", unsafe_allow_html=True)

# Main Form Layout
if model is not None:
    with st.form("prediction_form"):
        st.subheader("📋 Customer Information")
        
        # Gender Selection
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
        
        # Age Slider/Input
        age = st.slider("Age (Years)", min_value=18, max_value=100, value=35, step=1)
        
        # Salary Input
        salary = st.number_input("Estimated Annual Salary ($)", min_value=0, max_value=500000, value=50000, step=1000)
        
        # Submit Button
        submit_button = st.form_submit_button(label="Analyze Customer Profile", use_container_width=True)
        
    if submit_button:
        # Preprocess input data to match expected shape and feature names
        # Mapping Gender to numerical values if required by your model (commonly Male=1, Female=0 or vice versa)
        # Note: If your model was trained on strings directly, remove the mapping below.
        gender_encoded = 1 if gender == "Male" else 0
        
        # Create a DataFrame to pass to the model (preserves feature names if model needs them)
        input_data = pd.DataFrame([{
            'Gender': gender_encoded,
            'Age': age,
            'EstimatedSalary': salary
        }])
        
        # Make Prediction
        try:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            st.write("---")
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-box positive-result">
                        🎉 <strong>High Likelihood!</strong> This customer is likely to purchase. <br>
                        <span style="font-size: 0.9rem; font-weight: normal;">Confidence: {probabilities[1]*100:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-box negative-result">
                        ❌ <strong>Low Likelihood.</strong> This customer is unlikely to purchase. <br>
                        <span style="font-size: 0.9rem; font-weight: normal;">Confidence: {probabilities[0]*100:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Check if your model requires raw string values or custom encoded features for 'Gender'.")
