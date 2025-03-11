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
    output = model.predict([inputs])[0]
    return output

# Function to analyze symptoms using NLP
def analyze_symptoms(symptom_text):
    # Expanded symptom mapping
    symptoms_map = {
        'fatigue': 1,  # Hypothyroid
        'weight gain': 1,  # Hypothyroid
        'dry skin': 1,  # Hypothyroid
        'cold intolerance': 1,  # Hypothyroid
        'constipation': 1,  # Hypothyroid
        'weight loss': 2,  # Hyperthyroid
        'nervousness': 2,  # Hyperthyroid
        'rapid heartbeat': 2,  # Hyperthyroid
        'sweating': 2,  # Hyperthyroid
        'heat intolerance': 2,  # Hyperthyroid
    }

    detected_conditions = set()

    # Basic symptom analysis by keyword matching
    symptom_text_cleaned = re.sub(r'[^\w\s]', '', symptom_text.lower())
    for symptom, condition in symptoms_map.items():
        if symptom in symptom_text_cleaned:
            detected_conditions.add(condition)

    return detected_conditions

# Streamlit app
def main():
    # Title
    st.markdown(title_css, unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("<h3 style='color: #F63366;'>Sections</h3>", unsafe_allow_html=True)
    st.sidebar.write("1. About")
    st.sidebar.write("2. Instructions")
    st.sidebar.write("3. Contact")

    # **About Section with Thyroid Disorder Details**
    st.sidebar.title("About Thyroid Disorders")
    st.sidebar.write("""
        The **thyroid gland** produces hormones that regulate metabolism, energy, and overall body function.
        
        There are **two primary disorders**:
        
        **1. Hypothyroidism (Underactive Thyroid)**
        - Symptoms: Fatigue, weight gain, dry skin, cold intolerance, constipation.
        - Common Causes: Hashimoto's disease, iodine deficiency.
        
        **2. Hyperthyroidism (Overactive Thyroid)**
        - Symptoms: Weight loss, anxiety, sweating, heat intolerance, rapid heartbeat.
        - Common Causes: Graves' disease, thyroid nodules.

        **Thyroid Function Test Ranges:**
        
        - **TSH (Thyroid Stimulating Hormone)**
          - Normal: **0.4 - 4.0 μIU/mL**
          - High: **Hypothyroidism**
          - Low: **Hyperthyroidism**
        
        - **T3 (Triiodothyronine)**
          - Normal: **0.8 - 2.0 ng/mL**
          - Low: **Hypothyroidism**
          - High: **Hyperthyroidism**
        
        - **TT4 (Total Thyroxine)**
          - Normal: **5.0 - 12.0 μg/dL**
          - Low: **Hypothyroidism**
          - High: **Hyperthyroidism**
        
        - **T4U (Thyroxine Uptake)**
          - Normal: **0.6 - 1.8**
        
        - **FTI (Free Thyroxine Index)**
          - Normal: **6.0 - 12.0**
          - Low: **Hypothyroidism**
          - High: **Hyperthyroidism**

        These tests help doctors determine the exact thyroid condition.
    """)

    # Sidebar Contact Info
    st.sidebar.title("Contact Developer")
    st.sidebar.write("<h1 style='color: #F63366; font-size: 36px;'>Shivam Yadav</h1>", unsafe_allow_html=True)
    st.sidebar.write("GitHub: [Shivam31817](https://github.com/Shivam31817)")
    st.sidebar.write("LinkedIn: [Shivam Yadav](https://www.linkedin.com/in/shivam-yadav-135642231/)")

    # Symptom input field
    symptom_text = st.text_area("Enter your symptoms (e.g., fatigue, anxiety, weight gain):", 
                                 help="Please list your symptoms separated by commas.")

    # Input fields (No structure changes, keeping as is)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', value=None)
        TSH = st.number_input('TSH', value=None)
    with col2:
        sex = st.selectbox('Sex', options=['', 'M', 'F'])
        T3 = st.number_input('T3', value=None)
    with col3:
        TT4 = st.number_input('TT4', value=None)
        FTI = st.number_input('FTI', value=None)

    # Predict button
    detect_button = st.button('Detect')
    if detect_button:
        inputs = preprocess_inputs(age, sex, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, TSH, T3, TT4, 0, FTI)
        diagnosis_num = predict_diagnosis(inputs)
        diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')
        st.markdown(f"<h1 style='text-align: center; color: {diagnosis_color};'>{diagnosis_label}</h1>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()

