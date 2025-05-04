import streamlit as st
from streamlit_chat import message
import pickle
import re
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Diagnosis mapping
diagnoses = {
    0: 'Negative',
    1: 'Hypothyroid',
    2: 'Hyperthyroid'
}

# Title and theme color
diagnosis_color = '#F63366'
title_color = '#F63366'
title_css = f"<h1 style='text-align: center; color: {title_color};'>Thyroid Diagnosis Predictor</h1>"

# Input preprocessing
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

# Predict using model
def predict_diagnosis(inputs):
    output = model.predict([inputs])[0]
    return output

# NLP-based symptom analyzer
def analyze_symptoms(symptom_text):
    symptoms_map = {
        'fatigue': 1, 'weight gain': 1, 'dry skin': 1,
        'cold intolerance': 1, 'constipation': 1,
        'weight loss': 2, 'nervousness': 2, 'rapid heartbeat': 2,
        'sweating': 2, 'heat intolerance': 2,
    }
    detected_conditions = set()
    symptom_text_cleaned = re.sub(r'[^\w\s]', '', symptom_text.lower())
    for symptom, condition in symptoms_map.items():
        if symptom in symptom_text_cleaned:
            detected_conditions.add(condition)
    return detected_conditions

# Chatbot function
def chatbot_response(user_message):
    # Simple predefined responses for demonstration
    responses = {
        "What is hypothyroidism?": "Hypothyroidism is a condition where the thyroid gland doesn't produce enough thyroid hormones, leading to symptoms like fatigue, weight gain, and dry skin.",
        "What is hyperthyroidism?": "Hyperthyroidism is when the thyroid gland is overactive and produces too much thyroid hormone, causing symptoms like weight loss, anxiety, and rapid heartbeat.",
        "What are thyroid tests?": "Thyroid tests measure the levels of thyroid hormones in your blood, such as TSH, T3, TT4, T4U, and FTI.",
    }
    # Return a default response if the question is not predefined
    return responses.get(user_message, "I'm sorry, I don't understand that question. Please try asking something else.")

# Main Streamlit app
def main():
    st.markdown(title_css, unsafe_allow_html=True)

    # Background
    st.markdown("""
    <style>
        .stApp {
            background-image: url('https://www.shutterstock.com/shutterstock/photos/2076134416/display_1500/stock-vector-endocrinologists-diagnose-and-treat-human-thyroid-gland-doctors-make-blood-test-on-hormones-2076134416.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            color: black;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("<h3 style='color: #F63366;'>Sections</h3>", unsafe_allow_html=True)
    st.sidebar.write("1. About")
    st.sidebar.write("2. Instructions")
    st.sidebar.write("3. Contact")
    st.sidebar.title("About Project :")
    st.sidebar.write("This Streamlit app serves as a Thyroid Diagnosis Predictor using machine learning and NLP-based symptom analysis.")

    # Symptom input
    symptom_text = st.text_area("Enter your symptoms (e.g., fatigue, anxiety, weight gain):",
                                 help="List symptoms separated by commas.")

    # Input form layout
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', value=None)
        query_on_thyroxine = st.selectbox('Query On Thyroxine', ['', 'No', 'Yes'])
        pregnant = st.selectbox('Pregnant', ['', 'No', 'Yes'])
        query_hypothyroid = st.selectbox('Query Hypothyroid', ['', 'No', 'Yes'])
        goitre = st.selectbox('Goitre', ['', 'No', 'Yes'])
        psych = st.selectbox('Psych', ['', 'No', 'Yes'])
        TT4 = st.number_input('TT4', value=None)

    with col2:
        sex = st.selectbox('Sex', ['', 'M', 'F'])
        on_antithyroid_meds = st.selectbox('On Antithyroid Meds', ['', 'No', 'Yes'])
        thyroid_surgery = st.selectbox('Thyroid Surgery', ['', 'No', 'Yes'])
        query_hyperthyroid = st.selectbox('Query Hyperthyroid', ['', 'No', 'Yes'])
        tumor = st.selectbox('Tumor', ['', 'No', 'Yes'])
        TSH = st.number_input('TSH', value=None)
        T4U = st.number_input('T4U', value=None)

    with col3:
        on_thyroxine = st.selectbox('On Thyroxine', ['', 'No', 'Yes'])
        sick = st.selectbox('Sick', ['', 'No', 'Yes'])
        I131_treatment = st.selectbox('I131 Treatment', ['', 'No', 'Yes'])
        lithium = st.selectbox('Lithium', ['', 'No', 'Yes'])
        hypopituitary = st.selectbox('Hypopituitary', ['', 'No', 'Yes'])
        T3 = st.number_input('T3', value=None)
        FTI = st.number_input('FTI', value=None)

    # Buttons
    with col2:
        detect_button = st.button('Detect')
        clear_button = st.button('Clear')

    if detect_button:
        with st.spinner("Making predictions..."):
            inputs = preprocess_inputs(age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick,
                                       pregnant, thyroid_surgery, I131_treatment, query_hypothyroid,
                                       query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych,
                                       TSH, T3, TT4, T4U, FTI)
            diagnosis_num = predict_diagnosis(inputs)
            diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')
            nlp_conditions = analyze_symptoms(symptom_text)
            nlp_diagnosis = ', '.join([diagnoses.get(cond, 'Unknown') for cond in nlp_conditions])

        # Display results (same as before)

    # Chatbot Section at the Bottom
    st.markdown(f"<h2 style='color: {diagnosis_color}; text-align: center;'>Ask the Chatbot</h2>", unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display the chat history
    for msg in st.session_state.messages:
        message(msg['content'], is_user=msg['is_user'])

    # Get user input
    user_message = st.text_input("Your question:")

    if user_message:
        st.session_state.messages.append({"content": user_message, "is_user": True})
        bot_message = chatbot_response(user_message)
        st.session_state.messages.append({"content": bot_message, "is_user": False})

if __name__ == '__main__':
    main()


