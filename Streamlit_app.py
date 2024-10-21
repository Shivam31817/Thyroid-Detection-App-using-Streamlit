import numpy as np
import streamlit as st
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define diagnosis mapping dictionary
diagnoses = {0: 'Negative', 1: 'Hypothyroid', 2: 'Hyperthyroid'}

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

    return [age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery, 
            I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, 
            psych, TSH, T3, TT4, T4U, FTI]

# Function to predict the diagnosis based on inputs
def predict_diagnosis(inputs):
    output = model.predict([inputs])[0]
    return output

# Symptom analysis based on user input
def analyze_symptoms(symptoms):
    hypothyroid_symptoms = ['Fatigue', 'Weight Gain', 'Dry Skin', 'Cold Intolerance', 'Constipation']
    hyperthyroid_symptoms = ['Weight Loss', 'Nervousness', 'Rapid Heartbeat', 'Sweating', 'Heat Intolerance']
    
    hypo_count = sum([1 for symptom in symptoms if symptom in hypothyroid_symptoms])
    hyper_count = sum([1 for symptom in symptoms if symptom in hyperthyroid_symptoms])

    if hypo_count > hyper_count:
        return "Based on the symptoms, the user might be at higher risk for **Hypothyroidism**."
    elif hyper_count > hypo_count:
        return "Based on the symptoms, the user might be at higher risk for **Hyperthyroidism**."
    else:
        return "The symptoms do not strongly indicate either Hypothyroidism or Hyperthyroidism."

# Streamlit app
def main():
    st.title("Thyroid Diagnosis Predictor with Symptom Analysis")

    # Sidebar for user input
    st.sidebar.title("Enter Patient Information")

    # Symptom Input Section
    st.sidebar.subheader("Symptom Analysis")
    selected_symptoms = st.sidebar.multiselect(
        "Select any symptoms you are experiencing",
        ['Fatigue', 'Weight Gain', 'Dry Skin', 'Cold Intolerance', 'Constipation', 
         'Weight Loss', 'Nervousness', 'Rapid Heartbeat', 'Sweating', 'Heat Intolerance']
    )

    # Analyze symptoms if any are selected
    if selected_symptoms:
        symptom_result = analyze_symptoms(selected_symptoms)
        st.sidebar.write(symptom_result)

    # Input fields with black colored labels
    st.sidebar.markdown('<span style="color:black;">Age:</span>', unsafe_allow_html=True)
    age = st.sidebar.number_input("", min_value=0, value=25)

    st.sidebar.markdown('<span style="color:black;">Sex:</span>', unsafe_allow_html=True)
    sex = st.sidebar.selectbox("", options=["", "M", "F"])

    st.sidebar.markdown('<span style="color:black;">On Thyroxine:</span>', unsafe_allow_html=True)
    on_thyroxine = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Query On Thyroxine:</span>', unsafe_allow_html=True)
    query_on_thyroxine = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">On Antithyroid Meds:</span>', unsafe_allow_html=True)
    on_antithyroid_meds = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Sick:</span>', unsafe_allow_html=True)
    sick = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Pregnant:</span>', unsafe_allow_html=True)
    pregnant = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Thyroid Surgery:</span>', unsafe_allow_html=True)
    thyroid_surgery = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">I131 Treatment:</span>', unsafe_allow_html=True)
    I131_treatment = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Query Hypothyroid:</span>', unsafe_allow_html=True)
    query_hypothyroid = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Query Hyperthyroid:</span>', unsafe_allow_html=True)
    query_hyperthyroid = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Lithium:</span>', unsafe_allow_html=True)
    lithium = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Goitre:</span>', unsafe_allow_html=True)
    goitre = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Tumor:</span>', unsafe_allow_html=True)
    tumor = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Hypopituitary:</span>', unsafe_allow_html=True)
    hypopituitary = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">Psych:</span>', unsafe_allow_html=True)
    psych = st.sidebar.selectbox("", options=["", "Yes", "No"])

    st.sidebar.markdown('<span style="color:black;">TSH level:</span>', unsafe_allow_html=True)
    TSH = st.sidebar.number_input("", value=1.0)

    st.sidebar.markdown('<span style="color:black;">T3 level:</span>', unsafe_allow_html=True)
    T3 = st.sidebar.number_input("", value=1.0)

    st.sidebar.markdown('<span style="color:black;">TT4 level:</span>', unsafe_allow_html=True)
    TT4 = st.sidebar.number_input("", value=1.0)

    st.sidebar.markdown('<span style="color:black;">T4U level:</span>', unsafe_allow_html=True)
    T4U = st.sidebar.number_input("", value=1.0)

    st.sidebar.markdown('<span style="color:black;">FTI level:</span>', unsafe_allow_html=True)
    FTI = st.sidebar.number_input("", value=1.0)

    # Predict button
    if st.sidebar.button("Detect"):
        inputs = preprocess_inputs(age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant,
                                   thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium,
                                   goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI)

        # Predict the diagnosis
        diagnosis_num = predict_diagnosis(inputs)
        diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')

        # Display diagnosis
        st.write(f"### ML Diagnosis: {diagnosis_label}")

if __name__ == '__main__':
    main()







