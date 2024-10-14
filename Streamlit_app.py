import streamlit as st
import pickle
from PIL import Image

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define diagnosis mapping dictionary
diagnoses = {
    0: 'Negative',
    1: 'Hypothyroid',
    2: 'Hyperthyroid'
}

# Colors for diagnosis display
diagnosis_color = '#F63366'
title_color = '#F63366'  # Title color
title_css = f"<h1 style='text-align: center; color: {title_color};'>Thyroid Diagnosis Predictor</h1>"

# Button color
detect_button_color = '#F63366'

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
    # Load a local image for background
    image = Image.open('path_to_your_image.jpg')  # Specify the path to your local image file
    
    # Title
    st.markdown(title_css, unsafe_allow_html=True)

    # Add background image
    st.image(image, use_column_width=True, caption='', output_format='auto')

    # Sidebar
    st.sidebar.write("<h1 style='color: #F63366; font-size: 36px;'>Shivam Yadav</h1>", unsafe_allow_html=True)
    st.sidebar.write("GitHub profile: [link](https://github.com/Shivam31817)")
    st.sidebar.write("LinkedIn profile: [link](https://www.linkedin.com/in/shivam-yadav-135642231/)")

    st.sidebar.title("About Project:")
    st.sidebar.write("This Streamlit app serves as a Thyroid Diagnosis Predictor. It utilizes machine learning to predict thyroid diagnosis based on various patient attributes such as age, sex, medical history, and laboratory test results. Users can input patient data and receive an immediate diagnosis prediction, helping medical professionals make informed decisions efficiently.")

    st.sidebar.title("Attributes Information:")
    st.sidebar.write("""
        - Age: Age of the patient (int)
        - Sex: Sex patient identifies (str)
        - On Thyroxine: Whether patient is on thyroxine (bool)
        - Query on Thyroxine: Whether patient is on thyroxine (bool)
        - On Antithyroid Meds: Whether patient is on antithyroid meds (bool)
        - Sick: Whether patient is sick (bool)
        - Pregnant: Whether patient is pregnant (bool)
        - Thyroid Surgery: Whether patient has undergone thyroid surgery (bool)
        - I131 Treatment: Whether patient is undergoing I131 treatment (bool)
        - Query Hypothyroid: Whether patient believes they have hypothyroid (bool)
        - Query Hyperthyroid: Whether patient believes they have hyperthyroid (bool)
        - Lithium: Whether patient takes lithium (bool)
        - Goitre: Whether patient has goitre (bool)
        - Tumor: Whether patient has tumor (bool)
        - Hypopituitary: Whether patient is hypopituitary (bool)
        - Psych: Whether patient is psych (bool)
        - TSH: TSH level in blood from lab work (float)
        - T3: T3 level in blood from lab work (float)
        - TT4: TT4 level in blood from lab work (float)
        - T4U: T4U level in blood from lab work (float)
        - FTI: FTI level in blood from lab work (float)
    """)

    # Input fields
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
        detect_button_container = st.container()
        with detect_button_container:
            detect_button_css = f"""
                <style>
                    .stButton > button:first-child {{
                        width: 100%;
                        color: white;
                        border-color: {detect_button_color};
                        border-radius: 5px;
                        padding: 10px;
                    }}
                </style>
            """
            st.markdown(detect_button_css, unsafe_allow_html=True)

        if detect_button:
            # Preprocess inputs
            inputs = preprocess_inputs(age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds)


