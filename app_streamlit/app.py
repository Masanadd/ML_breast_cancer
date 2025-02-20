import streamlit as st
import pandas as pd
import joblib

# Load the trained model
best_model_svm = joblib.load("../models/final_model.pkl")

# Theme configuration
PINK_THEME = {
    "primary": "#FF69B4",  # Pink for breast cancer awareness
    "secondary": "#FFB6C1",  # Light pink
    "background": "#FFF0F6",  # Very light pink
    "text": "#333333",  # Dark gray for readability
}

# Selected features for prediction
selected_features = [
    'Relapse Free Status (Months)', 'Age at Diagnosis', 'Tumor Size', 
    'Mutation Count', 'Aggressive Treatment Score', 'Nottingham prognostic index', 
    'Lymph nodes examined positive', 'Type of Breast Surgery', 'Tumor Stage'
]

# Prediction function
def predict_survival_probabilities(data):
    df = pd.DataFrame([data], columns=selected_features)
    probabilities = best_model_svm.predict_proba(df)[0]
    return probabilities[1], probabilities[0]  # Survival probability first

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Survival Predictor",
    layout="centered",
)


# Insertar el logo centrado correctamente
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo.png", width=300)



# Custom CSS for styling
st.markdown(f"""
<style>
    .stApp {{
        background-color: {PINK_THEME['background']};
    }}
    .stButton>button {{
        background-color: {PINK_THEME['primary']};
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stSlider>div>div>div>div {{
        background-color: {PINK_THEME['primary']};
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {PINK_THEME['text']};
    }}
    .result-card {{
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .progress-bar {{
        height: 10px;
        border-radius: 5px;
        background: {PINK_THEME['secondary']};
        margin-top: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# Header with ribbon
st.markdown(f"""
<div style="text-align: center;">
    <h1 style="color: {PINK_THEME['primary']}; font-size: 36px;">OS Predictor Extension</h1>
    <div style="height: 4px; width: 80px; background: {PINK_THEME['primary']}; margin: 10px auto; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)

# Input sections in columns
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("""
            <h3 style="color: #FF69B4;"> Patient Profile</h3>
            <div style="height: 2px; background-color: #FF69B4; margin: 10px 0;"></div>
        """, unsafe_allow_html=True)
        age_diagnosis = st.slider("Age at Diagnosis", 21, 96, 50, help="Patient's age at the time of diagnosis")
        relapse_months = st.slider("Relapse-Free Months", 0, 384, 30, help="Months since the last relapse")

with col2:
    with st.container():
        st.markdown("""
            <h3 style="color: #FF69B4;">Tumor Characteristics</h3>
            <div style="height: 2px; background-color: #FF69B4; margin: 10px 0;"></div>
        """, unsafe_allow_html=True)
        tumor_size = st.slider("Tumor Size (mm)", 1, 120, 30, help="Measured tumor diameter")
        tumor_stage = st.selectbox("Tumor Stage", [1, 2, 3, 4], format_func=lambda x: f"Stage {x}")

# Clinical factors section
with st.container():
    st.markdown("""
        <h3 style="color: #FF69B4;"> Clinical Factors</h3>
        <div style="height: 2px; background-color: #FF69B4; margin: 10px 0;"></div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        mutation_count = st.number_input("Mutation Count", 0, 200, 10)
        lymph_nodes_positive = st.number_input("Positive Lymph Nodes", 0, 15, 2)
    with c2:
        aggressive_treatment = st.slider("Treatment Aggressiveness", 0, 4, 2, help="0 = Conservative, 4 = Most Aggressive")
    with c3:
        nottingham_index = st.slider("Nottingham Index", 0.0, 7.5, 3.5, step=0.1)
        breast_surgery = st.radio("Surgery Type", ["Mastectomy", "Lumpectomy"])

# Prediction button
if st.button("Calculate Survival Probability", use_container_width=True):
    breast_surgery = 1 if breast_surgery == "Mastectomy" else 0
    user_data = [
        relapse_months, age_diagnosis, tumor_size, 
        mutation_count, aggressive_treatment, nottingham_index, 
        lymph_nodes_positive, breast_surgery, tumor_stage
    ]
    
    prob_survival, prob_death = predict_survival_probabilities(user_data)
    
    # Display results with improved design
    st.markdown(f"""
    <div class="result-card">
        <h3 style="color: {PINK_THEME['primary']}; margin-bottom: 10px;">📊 Prediction Results</h3>
        <p style="font-size: 18px;">✅ <b>Survival Probability:</b> {prob_survival * 100:.1f}%</p>
        <div class="progress-bar" style="width: {prob_survival * 100}%; background: {PINK_THEME['primary']};"></div>
        <p style="font-size: 18px;">❌ <b>Mortality Risk:</b> {prob_death * 100:.1f}%</p>
        <div class="progress-bar" style="width: {prob_death * 100}%; background: #666;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Clinical interpretation
    st.markdown(f"""
    <div class="result-card">
        <h4 style="color: {PINK_THEME['text']}; margin-bottom: 10px;">📌 Clinical Interpretation</h4>
        <p style="font-size: 16px;">
            {'High survival probability' if prob_survival > 0.7 else 'Moderate prognosis' if prob_survival > 0.4 else 'Critical prognosis'} based on input parameters.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("🔍 This tool is designed to assist healthcare professionals and should not replace clinical judgment.")