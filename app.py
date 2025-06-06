import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join('models', 'trained', 'best_model.pkl')

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Try to infer feature names from the model pipeline
try:
    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
        preprocessor = model.named_steps['preprocessor']
        if hasattr(preprocessor, 'feature_names_in_'):
            feature_names = list(preprocessor.feature_names_in_)
        else:
            # Fallback: try to get from transformers
            feature_names = []
            for name, trans, cols in preprocessor.transformers_:
                feature_names.extend(cols)
    else:
        feature_names = []
except Exception:
    feature_names = []

# If feature names cannot be inferred, ask user to input as CSV
st.title('🏠 House Price Prediction')
st.write('Enter the required features to predict house price.')

input_data = {}

if feature_names:
    for feature in feature_names:
        # For demo, use number input. You can customize based on your data.
        input_data[feature] = st.number_input(f"{feature}", value=0.0)
    input_df = pd.DataFrame([input_data])
else:
    st.warning('Could not infer feature names. Please upload a CSV file with the correct columns.')
    uploaded_file = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = None

if st.button('Predict'):
    if input_df is not None:
        try:
            prediction = model.predict(input_df)
            st.success(f'Predicted house price: {prediction[0]:,.0f}')
        except Exception as e:
            st.error(f'Prediction failed: {e}')
    else:
        st.error('Please provide input data.') 