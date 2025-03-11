import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF for PDF text extraction
import googlemaps
from PIL import Image
import os

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

gmaps = googlemaps.Client(key="YOUR_GOOGLE_MAPS_API_KEY")  # Replace with your API key

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
        'fatigue': 1, 'weight gain': 1, 'dry skin': 1, 'cold intolerance': 1, 'constipation': 1,
        'weight loss': 2, 'nervousness': 2, 'rapid heartbeat': 2, 'sweating': 2, 'heat intolerance': 2,
    }
    detected_conditions = set()
    symptom_text_cleaned = re.sub(r'[^\w\s]', '', symptom_text.lower())
    for symptom, condition in symptoms_map.items():
        if symptom in symptom_text_cleaned:
            detected_conditions.add(condition)
    return detected_conditions

# Function to extract text from image
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text")
    return text

# Function to auto-fill lab values
def autofill_lab_values(text):
    values = {}
    keywords = {"TSH": "TSH", "T3": "T3", "TT4": "TT4", "T4U": "T4U", "FTI": "FTI"}
    
    for key, label in keywords.items():
        for line in text.split("\n"):
            if label in line:
                parts = line.split()
                for part in parts:
                    try:
                        values[key] = float(part)
                        break
                    except ValueError:
                        continue
    return values

# Function to find nearby doctors
def find_nearby_doctors(location):
    places = gmaps.places_nearby(location=location, radius=5000, type="doctor", keyword="endocrinologist")
    return places.get("results", [])

# Streamlit app
def main():
    st.markdown(title_css, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Lab Report (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        extracted_text = ""
        
        if file_extension in ["png", "jpg", "jpeg"]:
            image = Image.open(uploaded_file)
            extracted_text = extract_text_from_image(image)
        elif file_extension == "pdf":
            pdf_path = f"temp_{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            extracted_text = extract_text_from_pdf(pdf_path)
            os.remove(pdf_path)
        
        lab_values = autofill_lab_values(extracted_text)
        if lab_values:
            st.success("Extracted Lab Values:")
            for key, value in lab_values.items():
                st.write(f"{key}: {value}")
        else:
            st.warning("Could not detect lab values automatically.")

    if st.button("Find an Endocrinologist Near Me"):
        location = "Your City, Your Country"  # Replace with dynamic user location if possible
        doctors = find_nearby_doctors(location)
        if doctors:
            st.success("Nearby Endocrinologists:")
            for doc in doctors[:5]:
                st.write(f"- **{doc['name']}**\n  Address: {doc['vicinity']}")
        else:
            st.error("No endocrinologists found nearby.")

if __name__ == '__main__':
    main()


