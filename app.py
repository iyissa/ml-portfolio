import streamlit as st
import joblib
import numpy as np

# Load the pre-trained models
binary_model = joblib.load('models/xgb_model_binary.pkl')
multi_class_model = joblib.load('models/xgb_model_multi.pkl')

# Define class labels for multi-class classification
failure_classes = {
    0: "No Failure",
    1: "Power Failure",
    2: "Overstrain Failure",
    3: "Heat Dissipation Failure",
    4: "Tool Wear Failure"
}

# Define function for predictions
def predict_failure(model, input_data, task_type):
    prediction = model.predict([input_data])[0]
    probabilities = model.predict_proba([input_data])[0]
    
    if task_type == "Binary Classification":
        result = "Yes" if prediction == 1 else "No"
        return f"Failure Prediction: {result}", probabilities
    
    elif task_type == "Multi-class Classification":
        sorted_indices = np.argsort(probabilities)[::-1]
        top_classes = [(failure_classes[i], probabilities[i]) for i in sorted_indices[:3]]
        return f"Predicted Failure Type: {failure_classes[prediction]}", top_classes

# Streamlit UI
st.title("Machine Failure Prediction")
st.write("Enter your machine parameters and select the prediction type.")

# Input fields
air_temp = st.number_input("Air Temperature", min_value=0.0, step=0.1)
process_temp = st.number_input("Process Temperature", min_value=0.0, step=0.1)
rot_speed = st.number_input("Rotational Speed", min_value=0, step=1)
torque = st.number_input("Torque", min_value=0.0, step=0.1)
tool_wear = st.number_input("Tool Wear", min_value=0, step=1)

# Task selection
task_type = st.selectbox("Select Prediction Type", ["Binary Classification", "Multi-Class Classification"])

# Predict button
if st.button("Predict"):
    input_data = [air_temp, process_temp, rot_speed, torque, tool_wear]
    
    if task_type == "Binary Classification":
        result, probabilities = predict_failure(binary_model, input_data, task_type)
        st.write(result)
        st.write(f"Probability of Failure: {probabilities[1]:.2f}")
    else:
        result, top_classes = predict_failure(multi_class_model, input_data, task_type)
        st.write(result)
        st.write("Top Failure Predictions:")
        for label, prob in top_classes:
            st.write(f"{label}: {prob:.2f}")
