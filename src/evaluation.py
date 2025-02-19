import numpy as np
import pandas as pd
import joblib

def evaluate_model(model_path="models/final_model.pkl"):
    """
    Genera datos sintéticos de pacientes de alto y bajo riesgo y evalúa el modelo LGBM.
    """
    features_selected = [
        'Relapse Free Status (Months)', 'Age at Diagnosis', 'Tumor Size', 
        'Mutation Count', 'Aggressive Treatment Score', 'Nottingham prognostic index', 
        'Lymph nodes examined positive', 'Type of Breast Surgery', 'Tumor Stage'
    ]
    
    best_model_lgbm = joblib.load(model_path)
    
    high_risk_patients = pd.DataFrame({
        'Relapse Free Status (Months)': np.random.randint(6, 24, size=5),  
        'Age at Diagnosis': np.random.randint(61, 80, size=5),
        'Tumor Size': np.random.uniform(5.1, 10.0, size=5),
        'Mutation Count': np.random.randint(51, 150, size=5),
        'Aggressive Treatment Score': np.random.randint(3, 5, size=5),
        'Nottingham prognostic index': np.random.uniform(5.1, 7.5, size=5),
        'Lymph nodes examined positive': np.random.randint(5, 15, size=5),
        'Type of Breast Surgery': np.random.choice([0, 1], size=5), 
        'Tumor Stage': np.random.randint(3, 4, size=5)  
    })
    
    low_risk_patients = pd.DataFrame({
        'Relapse Free Status (Months)': np.random.randint(36, 120, size=5),  
        'Age at Diagnosis': np.random.randint(30, 50, size=5),
        'Tumor Size': np.random.uniform(0.5, 2.0, size=5),
        'Mutation Count': np.random.randint(1, 10, size=5),
        'Aggressive Treatment Score': np.random.randint(0, 2, size=5),
        'Nottingham prognostic index': np.random.uniform(2.0, 3.5, size=5),
        'Lymph nodes examined positive': np.random.randint(0, 2, size=5),
        'Type of Breast Surgery': np.random.choice([0, 1], size=5),  
        'Tumor Stage': np.random.randint(1, 2, size=5)  
    })
    
    synthetic_patients = pd.concat([high_risk_patients, low_risk_patients], ignore_index=True)
    
    synthetic_patients = synthetic_patients[features_selected]
    
   
    y_pred = best_model_lgbm.predict(synthetic_patients)
    y_proba = best_model_lgbm.predict_proba(synthetic_patients)[:, 1] 
    
    
    synthetic_patients['Predicted Class'] = y_pred
    synthetic_patients['Recurrence Probability'] = y_proba
    
    
    return synthetic_patients

if __name__ == "__main__":
    results = evaluate_model()
    print(results)
