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
    
    # Sidebar info
    st.sidebar.write("<h1 style='color: #F63366; font-size: 36px;'>Shivam Yadav</h1>", unsafe_allow_html=True)
    st.sidebar.write("GitHub: [Shivam31817](https://github.com/Shivam31817)")
    st.sidebar.write("LinkedIn: [Shivam Yadav](https://www.linkedin.com/in/shivam-yadav-135642231/)")
    
    # About Section with Theory and Parameters
    st.sidebar.title("About Project")
    st.sidebar.write("""
        ### Thyroid Disorders:
        The thyroid gland plays a crucial role in regulating metabolism through the production of thyroid hormones.  
        **Common Thyroid Disorders:**  
        - **Hypothyroidism:** Underactive thyroid, leading to fatigue, weight gain, and cold intolerance.  
        - **Hyperthyroidism:** Overactive thyroid, causing weight loss, anxiety, and heat intolerance.  
    """)

    st.sidebar.write("""
        ### Thyroid Function Test Ranges:
        | Parameter  | Normal Range | Hypothyroidism | Hyperthyroidism |
        |-----------|-------------|---------------|---------------|
        | **TSH**   | 0.4 - 4.0 μIU/mL | > 4.5 μIU/mL (High) | < 0.3 μIU/mL (Low) |
        | **T3**    | 0.8 - 2.0 ng/mL  | < 0.8 ng/mL (Low) | > 2.0 ng/mL (High) |
        | **TT4**   | 5.0 - 12.0 μg/dL | < 5.0 μg/dL (Low) | > 12.0 μg/dL (High) |
        | **T4U**   | 0.6 - 1.8        | Normal or Low | Normal or High |
        | **FTI**   | 6.0 - 12.0       | < 6.0 (Low) | > 12.0 (High) |
    """)

    # Symptom input field
    symptom_text = st.text_area("Enter your symptoms (e.g., fatigue, anxiety, weight gain):", 
                                 help="Please list your symptoms separated by commas.")

    # Input fields for numeric data
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', value=None)
        TSH = st.number_input('TSH', value=None)
        TT4 = st.number_input('TT4', value=None)

    with col2:
        sex = st.selectbox('Sex', options=['', 'M', 'F'])
        T3 = st.number_input('T3', value=None)
        FTI = st.number_input('FTI', value=None)

    with col3:
        T4U = st.number_input('T4U', value=None)

    # Detect button
    if st.button('Detect'):
        # Show spinner while predicting
        with st.spinner("Making predictions..."):
            # Preprocess inputs
            inputs = preprocess_inputs(age, sex, None, None, None, None, None,
                                       None, None, None, None, None,
                                       None, None, None, None, TSH, T3, TT4, T4U, FTI)

            # Get prediction from ML model
            diagnosis_num = predict_diagnosis(inputs)
            diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')

            # Display diagnosis
            st.markdown(f"<div style='background-color: {diagnosis_color}; padding: 20px; border-radius: 10px;'>"
                        f"<h1 style='text-align: center; color: white;'>Diagnosis: {diagnosis_label}</h1>"
                        "</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()

