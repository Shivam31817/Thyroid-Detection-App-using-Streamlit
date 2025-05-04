import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import numpy as np
from streamlit_chat import message  # Make sure you have this import

# ... (your other code) ...

# --- Virtual Assistant Chatbot ---
def chatbot():
    st.subheader("Virtual Assistant")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for i, chat_message in enumerate(st.session_state["chat_history"]):
        message(chat_message["content"], is_user=chat_message["role"] == "user", key=f"chat_{i}")

    if prompt := st.chat_input("Ask me about symptoms, thyroid conditions, etc."):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Basic chatbot logic (can be significantly expanded)
        response = ""
        if "fatigue" in prompt.lower():
            response = "Fatigue is a common symptom associated with hypothyroidism."
        elif "weight loss" in prompt.lower():
            response = "Unexplained weight loss can be a symptom of hyperthyroidism."
        elif "TSH levels" in prompt.lower():
            response = "TSH (Thyroid Stimulating Hormone) levels are crucial for assessing thyroid function. Normal range is typically 0.4 - 4.0 μIU/mL."
        elif "hypothyroidism" in prompt.lower():
            response = "Hypothyroidism is a condition where the thyroid gland doesn't produce enough thyroid hormones. Common symptoms include fatigue, weight gain, and cold intolerance."
        elif "hyperthyroidism" in prompt.lower():
            response = "Hyperthyroidism is a condition where the thyroid gland produces too much thyroid hormone. Common symptoms include weight loss, nervousness, and rapid heartbeat."
        elif "goitre" in prompt.lower():
            response = "A goitre is an enlargement of the thyroid gland."
        elif "treatment" in prompt.lower() and "hypothyroid" in prompt.lower():
            response = "Hypothyroidism is often treated with thyroid hormone replacement medication like levothyroxine."
        elif "treatment" in prompt.lower() and "hyperthyroid" in prompt.lower():
            response = "Hyperthyroidism can be treated with medications, radioiodine therapy, or surgery."
        elif "symptoms of hypothyroid" in prompt.lower():
            response = "Common symptoms of hypothyroidism include fatigue, weight gain, dry skin, cold intolerance, and constipation."
        elif "symptoms of hyperthyroid" in prompt.lower():
            response = "Common symptoms of hyperthyroidism include weight loss, nervousness, rapid heartbeat, sweating, and heat intolerance."
        elif "T3" in prompt.upper() and "levels" in prompt.lower():
            response = "T3 (Triiodothyronine) is another important thyroid hormone. Normal range is typically 0.8 - 2.0 ng/mL."
        elif "T4" in prompt.upper() and "levels" in prompt.lower():
            response = "TT4 (Total Thyroxine) is a key thyroid hormone. Normal range is typically 5.0 - 12.0 μg/dL."
        else:
            response = "I can provide information about common thyroid symptoms and conditions. Please ask specific questions."

        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# --- Doctor Contact Section ---
def doctor_contact_info(predicted_diagnosis):
    st.subheader("Need to Consult a Doctor?")
    if predicted_diagnosis in sensitive_conditions or predicted_diagnosis == "Conflict":
        st.warning(f"The predicted diagnosis of '{predicted_diagnosis}' or a conflict in results suggests the need for medical consultation. Please find the contact information for doctors below:")
        for specialty, contact in doctor_contacts.items():
            st.write(f"**{specialty}:**")
            st.write(f"  - Phone: {contact[0]}")
            st.write(f"  - Email: {contact[1]}")
    else:
        st.info("For any concerns or further clarification, consider consulting a healthcare professional.")

# Streamlit app
def main():
    # Title
 # Title color
    title_color = '#F63366'
    # Title
    title_css = f"<h1 style='text-align: center; color: {title_color};'>Thyroid Diagnosis Predictor</h1>"
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

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("<h3 style='color: #F63366;'>Sections</h3>", unsafe_allow_html=True)
    st.sidebar.write("1. About")
    st.sidebar.write("2. Instructions")
    st.sidebar.write("3. Contact")
    st.sidebar.title("About Project :")
    st.sidebar.write("This Streamlit app serves as a Thyroid Diagnosis Predictor using machine learning and NLP-based symptom analysis.")
    st.sidebar.write("""
        The *thyroid gland* produces hormones that regulate metabolism, energy, and overall body function.

        There are *two primary disorders*:

        *1. Hypothyroidism (Underactive Thyroid)*
        - Symptoms: Fatigue, weight gain, dry skin, cold intolerance, constipation.
        - Common Causes: Hashimoto's disease, iodine deficiency.

        *2. Hyperthyroidism (Overactive Thyroid)*
        - Symptoms: Weight loss, anxiety, sweating, heat intolerance, rapid heartbeat.
        - Common Causes: Graves' disease, thyroid nodules.

        *Thyroid Function Test Ranges:*

        - *TSH (Thyroid Stimulating Hormone)*
          - Normal: *0.4 - 4.0 μIU/mL*
          - High: *Hypothyroidism*
          - Low: *Hyperthyroidism*

        - *T3 (Triiodothyronine)*
          - Normal: *0.8 - 2.0 ng/mL*
          - Low: *Hypothyroidism*
          - High: *Hyperthyroidism*

        - *TT4 (Total Thyroxine)*
          - Normal: *5.0 - 12.0 μg/dL*
          - Low: *Hypothyroidism*
          - High: *Hyperthyroidism*

        - *T4U (Thyroxine Uptake)*
          - Normal: *0.6 - 1.8*

        - *FTI (Free Thyroxine Index)*
          - Normal: *6.0 - 12.0*
          - Low: *Hypothyroidism*
          - High: *Hyperthyroidism*

        These tests help doctors determine the exact thyroid condition.
    """)

    # Sidebar info
    st.sidebar.write("<h1 style='color: #F63366; font-size: 36px;'>Shivam Yadav</h1>", unsafe_allow_html=True)
    st.sidebar.write("GitHub: [Shivam31817](https://github.com/Shivam31817)")
    st.sidebar.write("LinkedIn: [Shivam Yadav](https://www.linkedin.com/in/shivam-yadav-135642231/)")


    # Symptom input field
    symptom_text = st.text_area("Enter your symptoms (e.g., fatigue, anxiety, weight gain):",
                                    help="Please list your symptoms separated by commas.")

    # Input fields for numeric data
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', value=None, help="Enter your age.")
        query_on_thyroxine = st.selectbox('Query On Thyroxine', options=['', 'No', 'Yes'],
                                            help="Is there a query about thyroxine?")
        pregnant = st.selectbox('Pregnant', options=['', 'No', 'Yes'], help="Are you pregnant?")
        query_hypothyroid = st.selectbox('Query Hypothyroid', options=['', 'No', 'Yes'],
                                            help="Is there a query about hypothyroidism?")
        goitre = st.selectbox('Goitre', options=['', 'No', 'Yes'], help="Do you have goitre?")
        psych = st.selectbox('Psych', options=['', 'No', 'Yes'], help="Do you have a psychological condition?")
        TT4 = st.number_input('TT4', value=None, help="Enter your TT4 level.")

    with col2:
        sex = st.selectbox('Sex', options=['', 'M', 'F'], help="Select your gender.")
        on_antithyroid_meds = st.selectbox('On Antithyroid Meds', options=['', 'No', 'Yes'],
                                                help="Are you on antithyroid medications?")
        thyroid_surgery = st.selectbox('Thyroid Surgery', options=['', 'No', 'Yes'],
                                            help="Have you had thyroid surgery?")
        query_hyperthyroid = st.selectbox('Query Hyperthyroid', options=['', 'No', 'Yes'],
                                                help="Is there a query about hyperthyroidism?")
        tumor = st.selectbox('Tumor', options=['', 'No', 'Yes'], help="Do you have any tumors?")
        TSH = st.number_input('TSH', value=None, help="Enter your TSH level.")
        T4U = st.number_input('T4U', value=None, help="Enter your T4U level.")

    with col3:
        on_thyroxine = st.selectbox('On Thyroxine', options=['', 'No', 'Yes'],
                                        help="Are you currently on thyroxine?")
        sick = st.selectbox('Sick', options=['', 'No', 'Yes'], help="Are you currently sick?")
        I131_treatment = st.selectbox('I131 Treatment', options=['', 'No', 'Yes'],
                                            help="Have you undergone I131 treatment?")
        lithium = st.selectbox('Lithium', options=['', 'No', 'Yes'], help="Are you taking lithium?")
        hypopituitary = st.selectbox('Hypopituitary', options=['', 'No', 'Yes'],
                                        help="Do you have hypopituitarism?")
        T3 = st.number_input('T3', value=None, help="Enter your T3 level.")
        FTI = st.number_input('FTI', value=None, help="Enter your FTI level.")

    st.markdown("---")
    with st.container():
        col_detect, col_chatbot = st.columns([2, 1])
        with col_detect:
            detect_button = st.button('Detect', key='predict_button')
            clear_button = st.button('Clear', key='clear_button')

            if detect_button:
                if model is None:
                    st.error("The machine learning model could not be loaded. Please check the file path.")
                else:
                    with st.spinner("Making predictions..."):
                        inputs = preprocess_inputs(age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick,
                                                    pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid,
                                                    lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI)

                        # Get prediction from ML model
                        diagnosis_num = predict_diagnosis(inputs)
                        if diagnosis_num is not None:
                            diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')

                            # Analyze symptoms using NLP
                            nlp_conditions = analyze_symptoms(symptom_text)
                            nlp_diagnosis = ', '.join([diagnoses.get(cond, 'Unknown') for cond in nlp_conditions])

                            # Conflict resolution: Check if NLP and ML contradict each other
                            if (1 in nlp_conditions and diagnosis_num == 2) or (2 in nlp_conditions and diagnosis_num == 1):
                                st.markdown(
                                    f"<div style='background-color: orange; padding: 15px; border-radius: 10px;'>"
                                    f"<h2 style='text-align: center; color: white;'>⚠️ Conflict Detected!</h2>"
                                    f"<p style='text-align: center; color: white;'>ML Diagnosis: <b>{diagnosis_label}</b></p>"
                                    f"<p style='text-align: center; color: white;'>NLP Suggested Diagnosis: <b>{nlp_diagnosis}</b></p>"
                                    f"<p style='text-align: center; color: white;'>The system detected conflicting results. Please consult a doctor for confirmation.</p>"
                                    "</div>", unsafe_allow_html=True
                                )
                                doctor_contact_info("Conflict") # Show doctor info in case of conflict
                            else:
                                # Display diagnosis normally if no conflict
                                st.markdown(f"<div style='background-color: {diagnosis_color}; padding: 20px; border-radius: 10px;'>"
                                            f"<h1 style='text-align: center; color: white;'>ML Diagnosis: {diagnosis_label}</h1>"
                                            "</div>", unsafe_allow_html=True)
                                doctor_contact_info(diagnosis_label) # Show doctor info based on ML diagnosis

                                if nlp_diagnosis:
                                    st.markdown(f"<div style='background-color: {diagnosis_color}; padding: 20px; border-radius: 10px;'>"
                                                f"<h2 style='text-align: center; color: white;'>NLP Suggested Diagnosis: {nlp_diagnosis}</h2>"
                                                "</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<h2 style='text-align: center; color: {diagnosis_color};'>No specific conditions detected from symptoms</h2>", unsafe_allow_html=True)

        with col_chatbot:
            chatbot()

        if clear_button:
            # Clear all input fields and chat history
            for key in st.session_state.keys():
                if key not in ['_is_rerun', '_session_id', '_streamlit_script_run_count', 'chat_history']:
                    del st.session_state[key]
            st.session_state['chat_history'] = []
            st.experimental_rerun()

if __name__ == '__main__':
    main()
