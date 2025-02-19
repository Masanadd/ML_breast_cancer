import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def load_test_data(test_path="data/test"):
    """
    Carga los datos de prueba desde la carpeta especificada.
    """
    X_test = pd.read_csv(os.path.join(test_path, "test_features.csv"))
    y_test = pd.read_csv(os.path.join(test_path, "test_labels.csv"))
    return X_test, y_test.squeeze()

def load_model(model_path="models/best_model.pkl"):
    """
    Carga el modelo entrenado desde la ubicación especificada.
    """
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test, output_path="data/test"):
    """
    Evalúa el modelo utilizando los datos de prueba y guarda las métricas de evaluación.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
 
    results = pd.DataFrame({
        "Métrica": ["Accuracy", "F1-Score", "ROC-AUC"],
        "Valor": [accuracy, f1, roc_auc]
    })
    report_df = pd.DataFrame(report).transpose()
    
    results.to_csv(os.path.join(output_path, "evaluation_metrics.csv"), index=False)
    report_df.to_csv(os.path.join(output_path, "classification_report.csv"), index=True)
    
    print("Evaluación finalizada. Métricas guardadas en data/test/")
    return results, report_df

if __name__ == "__main__":
    X_test, y_test = load_test_data()
    model = load_model()
    evaluate_model(model, X_test, y_test)
