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

# Detect button color
detect_button_color = '#F63366'

# Initialize a simple NLP model for symptom classification
vectorizer = TfidfVectorizer()
nlp_model = LogisticRegression()

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

    # Add custom CSS for background image
    background_image = """
    <style>
        .stApp {
            background-image: url('https://www.shutterstock.com/shutterstock/photos/2076134416/display_1500/stock-vector-endocrinologists-diagnose-and-treat-human-thyroid-gland-doctors-make-blood-test-on-hormones-2076134416.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            color: black;  /* Change text color to black for better visibility */
        }
    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.write("<h1 style='color: #F63366; font-size: 36px;'>Shivam Yadav</h1>", unsafe_allow_html=True)
    st.sidebar.write("GitHub profile: (https://github.com/Shivam31817)")
    st.sidebar.write("LinkedIn profile: (https://www.linkedin.com/in/shivam-yadav-135642231/)")
    
    st.sidebar.title("About Project :")
    st.sidebar.write("This Streamlit app serves as a Thyroid Diagnosis Predictor using machine learning and NLP-based symptom analysis.")

    # Symptom input field
    symptom_text = st.text_area("Enter your symptoms (e.g., fatigue, anxiety, weight gain):")

    # Input fields for numeric data
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', value=None)
        query_on_thyroxine = st.selectbox('Query On Thyroxine', options=['', 'No', 'Yes'])
        pregnant = st.selectbox('Pregnant', options=['', 'No', 'Yes'])
        query_hypothyroid = st.selectbox('Query Hypothyroid', options=['', 'No', 'Yes'])
        goitre = st.selectbox('Goitre', options=['', 'No', 'Yes'])
        psych = st.selectbox('Psych', options=['', 'No', 'Yes'])
        TT4 = st.number_input('TT4', value=None)

    with col2:
        sex = st.selectbox('Sex', options=['', 'M', 'F'])
        on_antithyroid_meds = st.selectbox('On Antithyroid Meds', options=['', 'No', 'Yes'])
        thyroid_surgery = st.selectbox('Thyroid Surgery', options=['', 'No', 'Yes'])
        query_hyperthyroid = st.selectbox('Query Hyperthyroid', options=['', 'No', 'Yes'])
        tumor = st.selectbox('Tumor', options=['', 'No', 'Yes'])
        TSH = st.number_input('TSH', value=None)
        T4U = st.number_input('T4U', value=None)

    with col3:
        on_thyroxine = st.selectbox('On Thyroxine', options=['', 'No', 'Yes'])
        sick = st.selectbox('Sick', options=['', 'No', 'Yes'])
        I131_treatment = st.selectbox('I131 Treatment', options=['', 'No', 'Yes'])
        lithium = st.selectbox('Lithium', options=['', 'No', 'Yes'])
        hypopituitary = st.selectbox('Hypopituitary', options=['', 'No', 'Yes'])
        T3 = st.number_input('T3', value=None)
        FTI = st.number_input('FTI', value=None)

    # Detect button
    with col2:
        detect_button = st.button('Detect', key='predict_button')
        if detect_button:
            # Preprocess inputs
            inputs = preprocess_inputs(age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick,
                                       pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid,
                                       lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI)

            # Get prediction from ML model
            diagnosis_num = predict_diagnosis(inputs)
            diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')

            # Analyze symptoms using NLP
            nlp_conditions = analyze_symptoms(symptom_text)
            nlp_diagnosis = ', '.join([diagnoses.get(cond, 'Unknown') for cond in nlp_conditions])

            # Display diagnosis
            st.markdown(f"<h1 style='text-align: center; color: {diagnosis_color};'>ML Diagnosis: {diagnosis_label}</h1>", unsafe_allow_html=True)

            if nlp_diagnosis:
                st.markdown(f"<h2 style='text-align: center; color: {diagnosis_color};'>NLP Suggested Diagnosis: {nlp_diagnosis}</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='text-align: center; color: {diagnosis_color};'>No specific conditions detected from symptoms</h2>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()









