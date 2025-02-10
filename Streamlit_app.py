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

# Function to preprocess inputs before prediction
def preprocess_inputs(age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant,
                      thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium,
                      goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI):
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

    sex = 1 if sex == 'F' else 0 if sex == 'M' else None

    return [age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant,
            thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium,
            goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI]

# Function to predict the diagnosis based on inputs
def predict_diagnosis(inputs):
    output = model.predict([inputs])[0]
    return output

# Function to analyze symptoms using NLP
def analyze_symptoms(symptom_text):
    symptoms_map = {
        'fatigue': 1,
        'weight gain': 1,
        'dry skin': 1,
        'cold intolerance': 1,
        'constipation': 1,
        'weight loss': 2,
        'nervousness': 2,
        'rapid heartbeat': 2,
        'sweating': 2,
        'heat intolerance': 2,
    }
    detected_conditions = set()
    symptom_text_cleaned = re.sub(r'[^\w\s]', '', symptom_text.lower())
    for symptom, condition in symptoms_map.items():
        if symptom in symptom_text_cleaned:
            detected_conditions.add(condition)
    return detected_conditions

# Streamlit app
def main():
    st.markdown(title_css, unsafe_allow_html=True)

    # Symptom input field
    symptom_text = st.text_area("Enter your symptoms (e.g., fatigue, anxiety, weight gain):", 
                                 help="Please list your symptoms separated by commas.")
    
    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', value=None, help="Enter your age.")
    with col2:
        sex = st.selectbox('Sex', options=['', 'M', 'F'], help="Select your gender.")
    with col3:
        on_thyroxine = st.selectbox('On Thyroxine', options=['', 'No', 'Yes'], help="Are you currently on thyroxine?")
    
    detect_button = st.button('Detect', key='predict_button')
    clear_button = st.button('Clear', key='clear_button')
    
    if detect_button:
        with st.spinner("Making predictions..."):
            inputs = preprocess_inputs(age, sex, on_thyroxine, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '')
            diagnosis_num = predict_diagnosis(inputs)
            diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')
            nlp_conditions = analyze_symptoms(symptom_text)
            nlp_diagnosis = ', '.join([diagnoses.get(cond, 'Unknown') for cond in nlp_conditions])
            
            st.markdown(f"<div style='background-color: {diagnosis_color}; padding: 20px; border-radius: 10px;'>"
                        f"<h1 style='text-align: center; color: white;'>ML Diagnosis: {diagnosis_label}</h1>"
                        "</div>", unsafe_allow_html=True)
            
            if nlp_diagnosis:
                st.markdown(f"<div style='background-color: {diagnosis_color}; padding: 20px; border-radius: 10px;'>"
                            f"<h2 style='text-align: center; color: white;'>NLP Suggested Diagnosis: {nlp_diagnosis}</h2>"
                            "</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='text-align: center; color: {diagnosis_color};'>No specific conditions detected from symptoms</h2>", unsafe_allow_html=True)
    
    if clear_button:
        st.experimental_rerun()
    
    # Chatbot Section
    st.markdown("---")
    st.markdown("### Chatbot Assistant")
    chat_history = st.session_state.get("chat_history", [])
    user_input = st.text_input("Ask me anything about thyroid diseases:")
    
    if st.button("Send") and user_input:
        response = f"You asked: {user_input}. I'm still learning, but I recommend consulting a doctor for medical advice."
        chat_history.append(("You", user_input))
        chat_history.append(("Bot", response))
        st.session_state.chat_history = chat_history
    
    for sender, message in chat_history:
        st.markdown(f"**{sender}:** {message}")

if __name__ == '__main__':
    main()


