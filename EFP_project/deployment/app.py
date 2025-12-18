import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="sindhoorasuresh/ML-Project", filename="best_engine_failure_model_v1.joblib")
model = joblib.load(model_path)


# Streamlit UI for Machine Failure Prediction
st.title("Engine Failure Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
engine_rpm = st.number_input("Engine rpm)", min_value=250.0, max_value=400.0, value=298.0, step=0.1)
lub_oil_pres = st.number_input("Lub oil pressure", min_value=250.0, max_value=500.0, value=324.0, step=0.1)
fuel_pres= st.number_input("Fuel pressure", min_value=0, max_value=3000, value=1400)
coolant_pres = st.number_input("Coolant pressure", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
lub_oil_temp = st.number_input("Lub oil temp", min_value=0, max_value=300, value=10)
coolant_temp = st.number_input("Coolant temp", min_value=0, max_value=300, value=10)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pres,
    'Fuel pressure': fuel_pres,
    'Coolant pressure': coolant_pres,
    'Lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Engine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
