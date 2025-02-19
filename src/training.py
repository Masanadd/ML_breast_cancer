import numpy as np
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def load_processed_data(data_path="data/processed"):
    """
    Carga los datos preprocesados desde la carpeta especificada.
    """
    processed_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    df_list = [pd.read_csv(os.path.join(data_path, f)) for f in processed_files]
    return pd.concat(df_list, ignore_index=True)

def train_and_save_model(data_path="data/processed", model_path="models/best_model.pkl", train_path="data/train", test_path="data/test"):
    """
    Carga los datos procesados, entrena un modelo y guarda el modelo junto con los conjuntos de datos.
    """
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists("models"):
        os.makedirs("models")
    
    df = load_processed_data(data_path)
    

    var_elim_clasf = [
        'Cellularity', 'Chemotherapy', 'ER Status', 'Neoplasm Histologic Grade', 'HER2 Status',
        'Hormone Therapy', 'Inferred Menopausal State', 'Primary Tumor Laterality',
        'Overall Survival (Months)', 'Overall Survival Status', 'PR Status', 'Radio Therapy',
        'Oncotree Code_Otros', 'Pam50 + Claudin-low subtype_LumB', 'Pam50 + Claudin-low subtype_Otros',
        'Pam50 + Claudin-low subtype_claudin-low', 'Risk Index'
    ]
    
    df = df.drop(columns=var_elim_clasf, errors="ignore")
    y = df["Overall Survival Status"]
    X = df.drop(columns=["Overall Survival Status"], errors="ignore")
    
   
    feature_selected_clasf = [
        'Relapse Free Status (Months)', 'Age at Diagnosis', 'Tumor Size', 'Mutation Count',
        'Aggressive Treatment Score', 'Nottingham prognostic index',
        'Lymph nodes examined positive', 'Type of Breast Surgery', 'Tumor Stage'
    ]
    X = X[feature_selected_clasf]
    
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    
 
    param_grid = {
        "classifier__n_estimators": [120],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__bootstrap": [True, False],
        "classifier__max_features": ["sqrt", 3, 4] 
    }
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="roc_auc",
        verbose=2
    )
    
 
    grid_search.fit(X_train, y_train)
    best_model_rfc = grid_search.best_estimator_

    y_pred_best = best_model_rfc.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)
    f1_best = f1_score(y_test, y_pred_best)
    roc_auc_best = roc_auc_score(y_test, y_pred_best)
    report_best = classification_report(y_test, y_pred_best, output_dict=True)
    

    X_train.to_csv(os.path.join(train_path, "train_features.csv"), index=False)
    X_test.to_csv(os.path.join(test_path, "test_features.csv"), index=False)
    y_train.to_csv(os.path.join(train_path, "train_labels.csv"), index=False)
    y_test.to_csv(os.path.join(test_path, "test_labels.csv"), index=False)
   
    joblib.dump(best_model_rfc, model_path)
    print(f"Modelo entrenado y guardado en {model_path}")
  
    results_rfc = pd.DataFrame({
        "Métrica": ["Accuracy", "F1-Score", "ROC-AUC"],
        "Valor": [accuracy_best, f1_best, roc_auc_best]
    })
    report_rfc = pd.DataFrame(report_best).transpose()
    
    results_rfc.to_csv(os.path.join(test_path, "model_metrics.csv"), index=False)
    report_rfc.to_csv(os.path.join(test_path, "classification_report.csv"), index=True)
    
    print("Entrenamiento finalizado. Métricas guardadas en data/test/")

if __name__ == "__main__":
    train_and_save_model()
