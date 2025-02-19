import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer

def analyze_missing_values(df):
    """
    Analiza el porcentaje de valores nulos en cada columna del DataFrame.
    Devuelve un DataFrame ordenado con las variables y su porcentaje de valores nulos.
    """
    missing_values = df.isnull().sum() / len(df) * 100
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    missing_df = pd.DataFrame({
        "Variable": missing_values.index,
        "Porcentaje de valores nulos (%)": missing_values.values
    })
    return missing_df

def plot_missing_values(df):
    """
    Genera un mapa de calor para visualizar los valores nulos en el dataset.
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cmap=sns.color_palette("coolwarm", as_cmap=True), cbar=False, yticklabels=False)
    plt.title("Mapa de Valores Faltantes en el Dataset")
    plt.show()

def clean_dataset(df):
    """
    Elimina columnas irrelevantes del dataset.
    """
    columns_to_drop = ["Patient ID", "Sex", "Cohort"]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    return df

def impute_numeric_values(df):
    """
    Imputa valores faltantes en variables numéricas usando la mediana.
    """
    imputer_median = SimpleImputer(strategy="median")
    numeric_columns = ["Lymph nodes examined positive", "Nottingham prognostic index", "Mutation Count"]
    for col in numeric_columns:
        df[col] = imputer_median.fit_transform(df[[col]])
    return df

def impute_tumor_stage(df):
    """
    Imputa valores faltantes en la variable "Tumor Stage" en función del tamaño del tumor.
    """
    def impute_stage(row):
        if pd.isna(row["Tumor Stage"]):  
            if row["Tumor Size"] <= 20:  
                return 1  
            elif row["Tumor Size"] <= 50:  
                return 2  
            else:  
                return 3  
        return row["Tumor Stage"]  
    
    df["Tumor Stage"] = df.apply(impute_stage, axis=1)
    return df

def encode_categorical_variables(df):
    """
    Codifica variables categóricas e imputa valores faltantes usando la moda.
    """
    categorical_vars = ["Type of Breast Surgery", "Cellularity", "Chemotherapy", "Hormone Therapy", "Radio Therapy"]
    for var in categorical_vars:
        df[var] = df.groupby("Cancer Type Detailed")[var].transform(
            lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else "Unknown"
        )
    categorical_vars_extra = ["HER2 Status", "PR Status", "ER Status"]
    for var in categorical_vars_extra:
        df[var] = df.groupby("Cancer Type Detailed")[var].transform(
            lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else "Unknown"
        )
    return df

def plot_histograms(df, columns):
    """
    Genera histogramas de distribución para las variables seleccionadas.
    """
    for col in columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], bins=30, color="pink", alpha=0.8, kde=True)
        plt.title(f"Distribución de {col}", fontsize=14)
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.show()

def preprocess_data(df):
    """
    Ejecuta el pipeline completo de preprocesamiento de datos, que incluye:
    1. Eliminación de columnas irrelevantes.
    2. Imputación de valores faltantes en variables numéricas.
    3. Imputación de valores faltantes en la variable "Tumor Stage".
    4. Codificación de variables categóricas e imputación de valores faltantes.
    Devuelve el DataFrame limpio y listo para el modelado.
    """
    df = clean_dataset(df)
    df = impute_numeric_values(df)
    df = impute_tumor_stage(df)
    df = encode_categorical_variables(df)
    return df

def process_and_save_data(input_path="data/raw", output_path="data/processed"):
    """
    Carga los datos desde la carpeta `data/raw`, aplica el preprocesamiento y guarda los datos procesados en `data/processed`.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    raw_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    
    for file in raw_files:
        df = pd.read_csv(os.path.join(input_path, file))
        df_processed = preprocess_data(df)
        df_processed.to_csv(os.path.join(output_path, f"processed_{file}"), index=False)
        print(f"Archivo procesado y guardado: processed_{file}")
