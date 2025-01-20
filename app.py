import streamlit as st
import joblib
import pandas as pd

# Load the XGBoost models for binary and multi-class tasks
xgb_model_binary = joblib.load("./models/xgb_model_binary.pkl")
xgb_model_multi = joblib.load("./models/xgb_model_multi.pkl")

# Store models in a dictionary for easier access
models = {
    'XGBoost Binary Model': xgb_model_binary,
    'XGBoost Multi-Class Model': xgb_model_multi
}

# Set the title of the app
st.title("Machine Failure Prediction")

# Input features from the user
st.header("Input Features")
air_temperature = st.number_input("Air Temperature", value=0.0)
process_temperature = st.number_input("Process Temperature", value=0.0)
rotational_speed = st.number_input("Rotational Speed", value=0.0)
torque = st.number_input("Torque", value=0.0)
tool_wear = st.number_input("Tool Wear", value=0.0)

# Select the model
selected_model = st.selectbox("Select Model", list(models.keys()))

# Button for prediction
if st.button("Predict"):
    # Prepare the input data for the model
    input_data = pd.DataFrame([[air_temperature, process_temperature, rotational_speed, torque, tool_wear]],
                              columns=["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"])  # Match feature names
    
    try:
        # Make prediction
        model = models[selected_model]
        prediction = model.predict(input_data)
        
        # Output prediction
        if selected_model == 'XGBoost Binary Model':
            if prediction[0] == 0:
                st.success("Machine will not fail.")
            else:
                st.warning("Machine will fail.")
        else:
            # For the multi-class model, assuming the model returns the class directly
            failure_type = prediction[0]  # Adjust based on your output logic
            st.warning(f"Machine will fail. Type of failure: {failure_type}")

    except Exception as e:
        st.error(f"An error occurred: {e}")