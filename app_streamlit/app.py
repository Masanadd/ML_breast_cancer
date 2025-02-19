import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado con los mejores hiperparámetros
best_model_svm = joblib.load("../models/final_model.pkl")

# Definir las características seleccionadas
selected_features = [
    'Relapse Free Status (Months)', 'Age at Diagnosis', 'Tumor Size', 
    'Mutation Count', 'Aggressive Treatment Score', 'Nottingham prognostic index', 
    'Lymph nodes examined positive', 'Type of Breast Surgery', 'Tumor Stage'
]

# Función para realizar la predicción
def predict_survival_probabilities(data):
    df = pd.DataFrame([data], columns=selected_features)
    probabilities = best_model_svm.predict_proba(df)[0]
    return probabilities[1], probabilities[0]  # Probabilidad de supervivencia primero

# Interfaz en Streamlit
st.title("🔬 Predicción de Supervivencia en Cáncer de Mama")
st.write("Ingrese las características clínicas del paciente para obtener una predicción de probabilidad de supervivencia.")

# Datos del Paciente
with st.container():
    st.header("📌 Datos del Paciente")
    age_diagnosis = st.slider("Edad al Diagnóstico", 21, 96, 50)
    relapse_months = st.slider("Meses sin Recaída", 0, 384, 30)

# Características del Tumor
with st.container():
    st.header("🩺 Características del Tumor")
    tumor_size = st.slider("Tamaño del Tumor (mm)", 1, 120, 30)
    mutation_count = st.slider("Conteo de Mutaciones", 0, 200, 10)
    tumor_stage = st.selectbox("Etapa del Tumor", [1, 2, 3, 4])

# Factores Pronósticos
with st.container():
    st.header("⚕️ Factores Pronósticos")
    aggressive_treatment = st.slider("Puntaje de Tratamiento Agresivo", 0, 4, 2)
    nottingham_index = st.slider("Nottingham Prognostic Index", 0.0, 7.5, 3.5)
    lymph_nodes_positive = st.slider("Linfáticos Positivos", 0, 15, 2)
    breast_surgery = 1 if st.radio("Tipo de Cirugía de Mama", ["Mastectomía", "Conservadora"]) == "Mastectomía" else 0

# Botón de Predicción
if st.button("🔍 Predecir Supervivencia"):
    user_data = [
        relapse_months, age_diagnosis, tumor_size, 
        mutation_count, aggressive_treatment, nottingham_index, 
        lymph_nodes_positive, breast_surgery, tumor_stage
    ]
    
    prob_survival, prob_death = predict_survival_probabilities(user_data)
    
    st.markdown(
        f"""
        <div style='background-color: white; padding: 20px; border-radius: 10px;'>
            <h3>📊 Resultados de Predicción</h3>
            <p>✅ <b>Probabilidad de Supervivencia:</b> {prob_survival * 100:.2f}%</p>
            <div style='height: 20px; width: {prob_survival * 100:.2f}%; background-color: #4CAF50; border-radius: 10px;'></div>
            <p>❌ <b>Probabilidad de Fallecimiento:</b> {prob_death * 100:.2f}%</p>
            <div style='height: 20px; width: {prob_death * 100:.2f}%; background-color: #FF4C4C; border-radius: 10px;'></div>
        </div>
        """,
        unsafe_allow_html=True
    )

