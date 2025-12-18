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
engine_rpm = st.number_input("Engine rpm", min_value=61.0000, max_value=2239.0000, value=876.0, step=0.1)
lub_oil_pres = st.number_input("Lub oil pressure", min_value=0.003384, max_value=7.2655, value=2.9416, step=0.1)
fuel_pres= st.number_input("Fuel pressure", min_value=0.0031, max_value=21.1383, value=16.1938)
coolant_pres = st.number_input("Coolant pressure", min_value=0.0024, max_value=7.4785, value=2.4645, step=0.1)
lub_oil_temp = st.number_input("lub oil temp", min_value=71.3219, max_value=89.5807, value=77.6409)
coolant_temp = st.number_input("Coolant temp", min_value=61.6733, max_value=195.5279, value=82.4457)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pres,
    'Fuel pressure': fuel_pres,
    'Coolant pressure': coolant_pres,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Engine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
