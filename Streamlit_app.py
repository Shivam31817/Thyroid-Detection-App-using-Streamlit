import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define diagnosis mapping dictionary
diagnoses = {
    0: 'Negative',
    1: 'Hypothyroid',
    2: 'Hyperthyroid'
}

# Set colors for the app
diagnosis_color = '#F63366'
title_color = '#F63366'
detect_button_color = '#F63366'

# Healthy ranges for lab results (for visualization)
healthy_ranges = {
    'TSH': (0.4, 4.0),  # Normal TSH range
    'T3': (80, 200),    # Normal T3 range (ng/dL)
    'TT4': (5.0, 12.0), # Normal TT4 range (ug/dL)
    'T4U': (0.7, 1.8),  # Normal T4U range
    'FTI': (6.5, 12.5)  # Normal FTI range
}

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

# Function to predict the diagnosis and provide confidence score
def predict_diagnosis(inputs):
    prediction = model.predict([inputs])[0]
    prediction_proba = model.predict_proba([inputs])[0]  # Confidence score (probability)
    return prediction, prediction_proba

# Function to plot feature importance (if your model supports it)
def plot_feature_importance():
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_names = ['Age', 'Sex', 'On Thyroxine', 'Query On Thyroxine', 'On Antithyroid Meds', 'Sick', 'Pregnant',
                         'Thyroid Surgery', 'I131 Treatment', 'Query Hypothyroid', 'Query Hyperthyroid', 'Lithium',
                         'Goitre', 'Tumor', 'Hypopituitary', 'Psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        sns.barplot(y=feature_names, x=feature_importances, palette="viridis")
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        st.pyplot()

# Function to validate inputs
def validate_inputs(age, TSH, T3, TT4, T4U, FTI):
    if age < 0 or age > 120:
        return "Age must be between 0 and 120"
    if TSH < 0 or T3 < 0 or TT4 < 0 or T4U < 0 or FTI < 0:
        return "Lab values cannot be negative"
    return None

# Streamlit app
def main():
    # Title
    st.markdown(f"<h1 style='text-align: center; color: {title_color};'>Thyroid Diagnosis Predictor</h1>", unsafe_allow_html=True)

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', value=30, min_value=0, max_value=120)
        query_on_thyroxine = st.selectbox('Query On Thyroxine', options=['', 'No', 'Yes'])
        pregnant = st.selectbox('Pregnant', options=['', 'No', 'Yes'])
        query_hypothyroid = st.selectbox('Query Hypothyroid', options=['', 'No', 'Yes'])
        goitre = st.selectbox('Goitre', options=['', 'No', 'Yes'])
        psych = st.selectbox('Psych', options=['', 'No', 'Yes'])
        TT4 = st.number_input('TT4', value=8.0)

    with col2:
        sex = st.selectbox('Sex', options=['', 'M', 'F'])
        on_antithyroid_meds = st.selectbox('On Antithyroid Meds', options=['', 'No', 'Yes'])
        thyroid_surgery = st.selectbox('Thyroid Surgery', options=['', 'No', 'Yes'])
        query_hyperthyroid = st.selectbox('Query Hyperthyroid', options=['', 'No', 'Yes'])
        tumor = st.selectbox('Tumor', options=['', 'No', 'Yes'])
        TSH = st.number_input('TSH', value=2.5)
        T4U = st.number_input('T4U', value=1.0)

    with col3:
        on_thyroxine = st.selectbox('On Thyroxine', options=['', 'No', 'Yes'])
        sick = st.selectbox('Sick', options=['', 'No', 'Yes'])
        I131_treatment = st.selectbox('I131 Treatment', options=['', 'No', 'Yes'])
        lithium = st.selectbox('Lithium', options=['', 'No', 'Yes'])
        hypopituitary = st.selectbox('Hypopituitary', options=['', 'No', 'Yes'])
        T3 = st.number_input('T3', value=100.0)
        FTI = st.number_input('FTI', value=9.0)

    # Detect button
    if st.button('Detect'):
        # Validate inputs
        validation_error = validate_inputs(age, TSH, T3, TT4, T4U, FTI)
        if validation_error:
            st.error(validation_error)
        else:
            # Preprocess inputs
            inputs = preprocess_inputs(age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick,
                                       pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, 
                                       query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, 
                                       TSH, T3, TT4, T4U, FTI)

            # Get prediction and confidence score
            diagnosis_num, diagnosis_proba = predict_diagnosis(inputs)
            diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')

            # Display diagnosis and confidence
            st.markdown(f"<h1 style='text-align: center; color: {diagnosis_color};'>{diagnosis_label}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center;'>Confidence: {max(diagnosis_proba) * 100:.2f}%</h4>", unsafe_allow_html=True)

            # Explanation based on diagnosis
            if diagnosis_num == 1:
                st.info("Hypothyroid detected. Consult an endocrinologist for further evaluation.")
            elif diagnosis_num == 2:
                st.info("Hyperthyroid detected. Treatment may include antithyroid medication or surgery.")
            else:
                st.info("No thyroid disorder detected. If symptoms persist, consider further testing.")

            # Plot feature importance
            st.subheader("Feature Importance")
            plot_feature_importance()

            # Visualize lab results vs. healthy range
            st.subheader("Lab Results vs Healthy Range")
            fig, ax = plt.subplots()
            ax.bar(healthy_ranges.keys(), [TSH, T3, TT4, T4U, FTI], color='lightblue', label='Your Values')
            for i, (key, (low, high)) in enumerate(healthy_ranges.items()):
                ax.plot([i-0.4, i+0.4], [low, low], color='green', linestyle='--')
                ax.plot([i-0.4, i+0.4], [high, high], color='green', linestyle='--')
            ax.set_ylabel('Lab Values')
            ax.legend()
            st.pyplot(fig)

if __name__ == '__main__':
    main()





