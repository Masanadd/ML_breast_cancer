# Breast Cancer Prediction using Machine Learning

## Descripcion

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning para clasificacion de cancer de mama. Se utilizan datos clinicos para entrenar y evaluar modelos predictivos, con el fin de mejorar el diagnostico temprano y la toma de decisiones medicas.


## Objetivo

El objetivo principal de este proyecto es crear un modelo de prediccion del **status free survival (SFS)**. El **status free survival** es el periodo de tiempo en el que un paciente permanece libre de progresion de la enfermedad o de recurrencia tras el tratamiento inicial. A diferencia de la supervivencia global, que mide el tiempo total que un paciente sobrevive, el SFS se centra en la ausencia de eventos adversos relacionados con el cancer. Esto permite evaluar la efectividad de los tratamientos y proporcionar informacion clave para la toma de decisiones clinicas.

## Dataset

- **Fuente**: La base de datos "The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC)" es un proyecto Canadá-Reino Unido que contiene datos de secuenciación dirigida de 1,980 muestras primarias de cáncer de mama. Los datos clínicos y genómicos fueron descargados de cBioPortal.
- **Variables clave**: Descripcion de las caracteristicas principales, como tamaño del tumor, textura, uniformidad celular, etc.
- **Numero de muestras**: Total de casos en el conjunto de datos.
- **Preprocesamiento**: Metodos aplicados (limpieza, normalizacion, eliminacion de valores atipicos, etc.).

## Variables

- **Patient ID:** Un identificador único para cada paciente.
- **Age at Diagnosis:** La edad del paciente en el momento del diagnóstico.
- **Type of Breast Surgery:** El tipo de cirugía mamaria que recibió el paciente (por ejemplo, mastectomía, tumorectomía).
- **Cancer Type:** El tipo general de cáncer (como carcinoma ductal, carcinoma lobulillar).
- **Cancer Type Detailed:** Una descripción más específica del tipo de cáncer.
- **Cellularity:** El grado de celularidad del tumor, que indica la densidad de células cancerígenas presentes.
- **Chemotherapy:** Indica si el paciente recibió quimioterapia como parte de su tratamiento.
- **Pam50 + Claudin-low subtype:** Un subgrupo molecular del cáncer basado en la clasificación PAM50, que incluye información sobre la expresión génica y la clasificación de subtipos como luminal A, luminal B, HER2-enriquecido, entre otros.
- **Cohort:** El grupo o cohorte al que pertenece el paciente dentro del estudio.
- **ER status measured by IHC:** Estado del receptor de estrógeno medido por inmunohistoquímica (IHC), que puede ser positivo o negativo.
- **ER Status:** Simplificación del estado del receptor de estrógeno (positivo o negativo).
- **Neoplasm Histologic Grade:** El grado histológico del tumor, que indica qué tan anormales se ven las células cancerosas bajo el microscopio y qué tan probable es que crezcan y se diseminen.
- **HER2 status measured by SNP6:** Estado del receptor HER2 medido mediante análisis genético, como el de un microarray de SNP.
- **HER2 Status:** Simplificación del estado HER2 (positivo, negativo o indeterminado).
- **Tumor Other Histologic Subtype:** Subtipo histológico adicional del tumor, que proporciona detalles más granulares sobre la clasificación histológica.
- **Hormone Therapy:** Indica si el paciente recibió terapia hormonal como parte del tratamiento.
- **Inferred Menopausal State:** Estado menopáusico inferido del paciente (por ejemplo, premenopáusica o postmenopáusica).
- **Integrative Cluster:** Una clasificación molecular que agrupa tumores con características similares basadas en múltiples tipos de datos moleculares.
- **Primary Tumor Laterality:** La lateralidad del tumor primario, indicando si el tumor estaba en el seno izquierdo o derecho.
- **Lymph nodes examined positive:** Número de ganglios linfáticos examinados que resultaron positivos para células cancerígenas.
- **Mutation Count:** Número total de mutaciones identificadas en el análisis genético del tumor.
- **Nottingham prognostic index:** Un índice pronóstico calculado en función del tamaño del tumor, el grado histológico y el estado de los ganglios linfáticos, usado para predecir el resultado del cáncer de mama.
- **Oncotree Code:** Código del árbol oncológico que clasifica los cánceres en subtipos y categorías con base en características moleculares y clínicas.
- **Overall Survival (Months):** Tiempo total de supervivencia en meses desde el diagnóstico o inicio del tratamiento hasta la muerte o la última revisión.
- **Overall Survival Status:** Estado de supervivencia general (si el paciente sigue vivo o ha fallecido).
- **PR Status:** Estado del receptor de progesterona (positivo o negativo).
- **Radio Therapy:** Indica si el paciente recibió radioterapia como parte del tratamiento.
- **Relapse Free Status (Months):** Tiempo en meses en que el paciente permaneció libre de recaídas después del tratamiento inicial.
- **Relapse Free Status:** Indica si el paciente ha experimentado una recaída.
- **Sex:** Sexo del paciente (masculino, femenino u otro).
- **3-Gene classifier subtype:** Un subgrupo del cáncer basado en la expresión de tres genes específicos, utilizado para clasificar tumores.
- **Tumor Size:** Tamaño del tumor primario, generalmente en milímetros.
- **Tumor Stage:** Etapa del cáncer, que describe la extensión de la enfermedad (por ejemplo, etapa I, II, III o IV).
- **Patient's Vital Status:** Estado vital del paciente al momento del análisis (vivo o fallecido).