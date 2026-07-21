
import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('diabetes_scaler.pkl')

st.title("Early Diabetes Risk Prediction System")
st.write("Enter the patient's health information to check for diabetes risk.")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Check Risk"):
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    input_as_numpy = np.asarray(input_data).reshape(1,-1)

    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame(input_as_numpy, columns=columns)
    std_data = scaler.transform(input_df)

    prediction = model.predict(std_data)

    if prediction[0] == 0:
        st.success('The patient is unlikely to have diabetes.')
    else:
        st.error('Warning: The patient has a high risk of diabetes.')
