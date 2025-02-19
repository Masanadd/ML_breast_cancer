import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier

def train_lgbm_model(data_path="Data/processed", model_path="models/best_lgbm_model.pkl", results_path="results"):
    var_elim_clasf = [
        'Cellularity', 'Chemotherapy', 'ER Status',
        'Neoplasm Histologic Grade', 'HER2 Status', 'Hormone Therapy',
        'Inferred Menopausal State', 'Primary Tumor Laterality',
        'Overall Survival (Months)', 'Overall Survival Status', 'PR Status',
        'Radio Therapy', 'Oncotree Code_Otros', 'Pam50 + Claudin-low subtype_LumB',
        'Pam50 + Claudin-low subtype_Otros', 'Pam50 + Claudin-low subtype_claudin-low',
        'Risk Index'
    ]
    
    X_train = pd.read_csv(f"{data_path}/train/X_train_clasf.csv")
    y_train = pd.read_csv(f"{data_path}/train/y_train_clasf.csv").squeeze()
    X_test = pd.read_csv(f"{data_path}/test/X_test_clasf.csv")
    y_test = pd.read_csv(f"{data_path}/test/y_test_clasf.csv").squeeze()
    
    feature_selected_clasf = [
        'Relapse Free Status (Months)', 'Age at Diagnosis', 'Tumor Size', 'Mutation Count',
        'Aggressive Treatment Score', 'Nottingham prognostic index',
        'Lymph nodes examined positive', 'Type of Breast Surgery', 'Tumor Stage'
    ]
    
    X_train = X_train[feature_selected_clasf]
    X_test = X_test[feature_selected_clasf]
    
    pipeline_lgbm = Pipeline([
        ("scaler", StandardScaler()),  
        ("classifier", LGBMClassifier(random_state=42))
    ])
    
    param_grid_lgbm = {
        "classifier__n_estimators": [100, 200, 300],  
        "classifier__learning_rate": [0.01, 0.1, 0.2],  
        "classifier__max_depth": [3, 5, 7],  
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0]
    }
    
    grid_search_lgbm = GridSearchCV(
        estimator=pipeline_lgbm,
        param_grid=param_grid_lgbm,
        cv=5,
        n_jobs=-1,
        scoring="roc_auc",
        verbose=2
    )
    
    grid_search_lgbm.fit(X_train, y_train)
    
    best_model_lgbm = grid_search_lgbm.best_estimator_
    
    y_pred_lgbm = best_model_lgbm.predict(X_test)
    
    accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
    f1_lgbm = f1_score(y_test, y_pred_lgbm)
    roc_auc_lgbm = roc_auc_score(y_test, y_pred_lgbm)
    report_lgbm = classification_report(y_test, y_pred_lgbm, output_dict=True)
    
    joblib.dump(best_model_lgbm, model_path)
    
    results_lgbm = pd.DataFrame({
        "Métrica": ["Accuracy", "F1-Score", "ROC-AUC"],
        "Valor": [accuracy_lgbm, f1_lgbm, roc_auc_lgbm]
    })
    
    report_lgbm_df = pd.DataFrame(report_lgbm).transpose()
    
    results_lgbm.to_csv(f"{results_path}/test_results.csv", index=False)
    report_lgbm_df.to_csv(f"{results_path}/classification_report.csv", index=True)
    
    print("Modelo y reportes guardados correctamente.")

if __name__ == "__main__":
    train_lgbm_model()

