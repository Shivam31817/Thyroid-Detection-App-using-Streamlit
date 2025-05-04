import streamlit as st
import pickle
import numpy as np
import re
import openai
from streamlit_chat import message

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

diagnoses = {0: 'Negative', 1: 'Hypothyroid', 2: 'Hyperthyroid'}

def preprocess_inputs(...):  # Keep your existing implementation
    

def predict_diagnosis(inputs):
    return model.predict([inputs])[0]

def analyze_symptoms(symptom_text):
    symptoms_map = {
        'fatigue': 1, 'weight gain': 1, 'dry skin': 1,
        'cold intolerance': 1, 'constipation': 1,
        'weight loss': 2, 'nervousness': 2, 'rapid heartbeat': 2,
        'sweating': 2, 'heat intolerance': 2,
    }
    detected = set()
    cleaned = re.sub(r'[^\w\s]', '', symptom_text.lower())
    for symptom, label in symptoms_map.items():
        if symptom in cleaned:
            detected.add(label)
    return detected

# Page: Prediction
def thyroid_prediction_page():
    st.title("🩺 Thyroid Disease Prediction System")

    st.write("Enter your symptoms and lab values:")

    # Input collection
    col1, col2, col3 = st.columns(3)
    # Your same input logic here for collecting age, sex, TSH, etc...

    # Collect inputs and run prediction
    if st.button("Detect"):
        inputs = preprocess_inputs(...)  # Fill your args
        diagnosis_num = predict_diagnosis(inputs)
        diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')

        # NLP symptom analysis
        nlp_conditions = analyze_symptoms(st.text_area("Enter symptoms:"))
        nlp_diagnosis = ', '.join(diagnoses.get(c, 'Unknown') for c in nlp_conditions)

        if diagnosis_num in nlp_conditions:
            st.success(f"✅ ML & NLP agree: {diagnosis_label}")
        else:
            st.warning(f"⚠️ Conflict! ML says {diagnosis_label}, NLP says {nlp_diagnosis}")

# Page: Virtual Assistant
def virtual_assistant_page():
    st.title("🤖 Virtual Health Assistant")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "system", "content": "You're a thyroid health assistant. Provide guidance, not diagnosis."}
        ]

    user_input = st.text_input("Ask a health-related question:")

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.chat_messages
            )
            reply = response.choices[0].message["content"]
        except Exception:
            reply = "⚠️ Chat error. Check API key."

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    for msg in st.session_state.chat_messages[1:]:
        message(msg["content"], is_user=(msg["role"] == "user"))

# --- App Entry ---
st.set_page_config(page_title="Thyroid Diagnosis App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Thyroid Prediction", "Virtual Assistant"])

if page == "Thyroid Prediction":
    thyroid_prediction_page()
elif page == "Virtual Assistant":
    virtual_assistant_page()

st.markdown("---")
st.markdown("© 2025 Thyroid Health Assistant | Built with ❤️ using Streamlit & OpenAI")


