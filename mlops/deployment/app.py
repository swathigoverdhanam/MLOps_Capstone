import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="SwathiSusarapu/Capstone-Final/engine-model", filename="best_engine_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Engine Prediction
st.title("Engine Prediction App")
st.write("Please enter engine details")

# Collect user input
['Engine rpm','Lub oil pressure','Fuel pressure','Coolant pressure','lub oil temp','Coolant temp']
EngineRPM = st.number_input("Engine RPM", min_value=61, max_value=2239, value=790)
LubOilPressure = st.selectbox("Lub oil pressure", min_value=0, max_value=7, value=4)
FuelPressure = st.number_input("Fuel pressure", min_value=0, max_value=21, value=7)
CoolantPressure = st.number_input("Coolant pressure", min_value=0, max_value=7, value=2)
LubOilTemp = st.number_input("lub oil temp", min_value=71, max_value=89, value=77)
CoolantTemp = st.number_input("Coolant temp", min_value=61, max_value=195, value=78)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Engine rpm': EngineRPM,
    'Lub oil pressure': LubOilPressure,
    'Fuel pressure': FuelPressure,
    'Coolant pressure': CoolantPressure,
    'lub oil temp': LubOilTemp,
    'Coolant temp': CoolantTemp
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Not Fail" if prediction == 1 else "Will fail"
    st.write(f"Based on the information provided, the engine is likely to {result}.")
