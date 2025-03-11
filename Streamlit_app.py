import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define diagnosis mapping dictionary
diagnoses = {
    0: 'Negative',
    1: 'Hypothyroid',
    2: 'Hyperthyroid'
}

# Predicted diagnosis color
diagnosis_color = '#F63366'
title_color = '#F63366'  # Title color
title_css = f"<h1 style='text-align: center; color: {title_color};'>Thyroid Diagnosis Predictor</h1>"

# Function to get prediction confidence
def get_prediction_confidence(proba):
    return round(max(proba) * 100, 2)

# Function to check symptom severity
def check_symptom_severity(symptom_text):
    severity_keywords = {
        'severe': 'High',
        'extreme': 'High',
        'mild': 'Low',
        'moderate': 'Medium',
        'critical': 'High'
    }
    for keyword, severity in severity_keywords.items():
        if keyword in symptom_text.lower():
            return severity
    return 'Unknown'

# Function to give patient advice
def get_patient_advice(diagnosis_label):
    advice_map = {
        'Negative': 'Your results are normal. Maintain a healthy lifestyle and regular check-ups.',
        'Hypothyroid': 'Consider consulting an endocrinologist. Proper medication and dietary adjustments can help.',
        'Hyperthyroid': 'Consult a specialist as soon as possible. Treatment may include medication or other therapies.'
    }
    return advice_map.get(diagnosis_label, 'Consult a healthcare professional for further evaluation.')

# Function to preprocess inputs before prediction
def preprocess_inputs(age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant,
                      thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium,
                      goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI):

    # Replace 'Yes' with 1 and 'No' with 0
    binary_map = {'Yes': 1, 'No': 0, '': None}
    on_thyroxine = binary_map.get(on_thyroxine)
    query_on_thyroxine = binary_map.get(query_on_thyroxine)
    on_antithyroid_meds = binary_map.get(on_antithyroid_meds)
    sick = binary_map.get(sick)
    pregnant = binary_map.get(pregnant)
    thyroid_surgery = binary_map.get(thyroid_surgery)
    I131_treatment = binary_map.get(I131_treatment)
    query_hypothyroid = binary_map.get(query_hypothyroid)
    query_hyperthyroid = binary_map.get(query_hyperthyroid)
    lithium = binary_map.get(lithium)
    goitre = binary_map.get(goitre)
    tumor = binary_map.get(tumor)
    hypopituitary = binary_map.get(hypopituitary)
    psych = binary_map.get(psych)

    # Replace 'M' and 'F' with binary 0 and 1
    sex = 1 if sex == 'F' else 0 if sex == 'M' else None

    return [age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant,
            thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium,
            goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI]

# Function to predict the diagnosis based on inputs
def predict_diagnosis(inputs):
    proba = model.predict_proba([inputs])[0]
    output = model.predict([inputs])[0]
    confidence = get_prediction_confidence(proba)
    return output, confidence

# Streamlit app
def main():
    st.markdown(title_css, unsafe_allow_html=True)

    symptom_text = st.text_area("Enter your symptoms (e.g., fatigue, anxiety, weight gain):", 
                                 help="Please list your symptoms separated by commas.")
    
    detect_button = st.button('Detect', key='predict_button')
    
    if detect_button:
        with st.spinner("Making predictions..."):
            inputs = preprocess_inputs(age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick,
                                   pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid,
                                   lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI)

        diagnosis_num, confidence = predict_diagnosis(inputs)
        diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')

        severity = check_symptom_severity(symptom_text)
        patient_advice = get_patient_advice(diagnosis_label)
        
        st.markdown(f"<div style='background-color: {diagnosis_color}; padding: 20px; border-radius: 10px;'>"
                    f"<h1 style='text-align: center; color: white;'>ML Diagnosis: {diagnosis_label} ({confidence}% confidence)</h1>"
                    f"<h2 style='text-align: center; color: white;'>Symptom Severity: {severity}</h2>"
                    f"<h3 style='text-align: center; color: white;'>Advice: {patient_advice}</h3>"
                    "</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()

