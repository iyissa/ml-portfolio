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
    
    elif task_type == "Multi-Class Classification":
        sorted_indices = np.argsort(probabilities)[::-1]
        top_classes = [(failure_classes[i], probabilities[i]) for i in sorted_indices[:3]]
        return f"Predicted Failure Type: {failure_classes[prediction]}", top_classes

# Streamlit UI
st.title("Machine Failure Prediction")
st.markdown(
    """
    🚀 **Welcome to the Predictive Maintenance System!**  
    This tool is the outward-facing result of a machine learning project on **Predictive Machine Failure**.  
    Engineers can **input machine sensor data** and use this tool to predict potential failures.  

    **How It Works:**  
    - Enter machine parameters (air temperature, process temperature, rotational speed, torque, tool wear).  
    - Choose between **Binary Classification** (Failure: Yes/No) or **Multi-Class Classification** (Failure Type).  
    - Get **real-time predictions** to prevent costly machine breakdowns! 

    ⚠️ **Note:** Ensure all inputs are **non-zero** for accurate predictions.
    """
)
st.write("Enter your machine parameters and select the prediction type.")

# Input fields (Minimum value set to 0.01 to prevent zero inputs)
air_temp = st.number_input("Air Temperature", min_value=0.01, step=0.1)
process_temp = st.number_input("Process Temperature", min_value=0.01, step=0.1)
rot_speed = st.number_input("Rotational Speed", min_value=1, step=1)
torque = st.number_input("Torque", min_value=0.01, step=0.1)
tool_wear = st.number_input("Tool Wear", min_value=1, step=1)

# Task selection
task_type = st.selectbox("Select Prediction Type", ["Binary Classification", "Multi-Class Classification"])

# Predict button
if st.button("Predict"):
    input_data = [air_temp, process_temp, rot_speed, torque, tool_wear]
    
    # Check if any input is zero (though the UI prevents it, double-checking for safety)
    if any(val <= 0 for val in input_data):
        st.error("⚠️ Please enter **values greater than zero** for all inputs before proceeding.")
    else:
        if task_type == "Binary Classification":
            result, probabilities = predict_failure(binary_model, input_data, task_type)
            st.success(result)
            st.write(f"Probability of Failure: {probabilities[1]:.2f}")
        else:
            result, top_classes = predict_failure(multi_class_model, input_data, task_type)
            st.success(result)
            st.write("Top Failure Predictions:")
            for label, prob in top_classes:
                st.write(f"{label}: {prob:.2f}")


st.markdown(
    """
    ---
    📌 **Check out the full project on GitHub:**  
    [🔗 iyissa/ml-portfolio](https://github.com/iyissa/ml-portfolio)
    """,
    unsafe_allow_html=True,
)