# 🏠 API de Predicción de Precios de Vivienda (Machine Learning)

Este proyecto es una solución completa de **Data Science y Machine Learning** que predice el precio de mercado de propiedades inmobiliarias en España basándose en sus características (metros cuadrados, ubicación, habitaciones, etc.).

El sistema abarca desde la ingesta de datos brutos hasta el despliegue de una **API REST** funcional construida con **Flask**, capaz de realizar predicciones en tiempo real.

## 📋 Descripción del Proyecto

El objetivo principal fue desarrollar un modelo predictivo robusto y exponerlo como un microservicio. El flujo de trabajo incluyó:

1.  **Ingesta de Datos (ETL):** Recopilación y unificación de múltiples datasets provinciales (CSV).
2.  **Limpieza y Pre-procesamiento:** Tratamiento de valores nulos, conversión de tipos y eliminación de *outliers* (valores atípicos) para mejorar la calidad del modelo.
3.  **Entrenamiento (Machine Learning):** Desarrollo de un modelo de regresión (**Random Forest Regressor**) utilizando `scikit-learn`. Se aplicó *One-Hot Encoding* para variables categóricas y se logró una precisión (R²) sólida en el set de pruebas.
4.  **Despliegue (API):** Creación de una API web con **Flask** que carga el modelo entrenado y procesa peticiones JSON para devolver estimaciones de precios.

## 🛠️ Stack Tecnológico

* **Lenguaje:** Python 3.10+
* **Análisis de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (RandomForest, métricas R²)
* **Backend / API:** Flask
* **Serialización:** Joblib, JSON

## 📂 Estructura del Proyecto

```text
├── data/                   # Carpeta de almacenamiento de datos
│   ├── houses_*.csv        # Archivos de datos brutos por provincia
│   ├── combined_housing_data.csv # Dataset unificado
│   ├── cleaned_housing_data.csv  # Dataset limpio para entrenamiento
│   ├── housing_price_model.pkl   # Modelo entrenado (binario)
│   └── model_columns.json        # Metadatos de columnas del modelo
│
├── 01_combine_data.py      # Script: Unifica los CSVs dispersos
├── 02_clean_data.py        # Script: Limpieza y transformación de datos
├── 03_train_model.py       # Script: Entrenamiento y evaluación del modelo
├── app.py                  # Aplicación principal (Servidor API Flask)
├── test_api.py             # Script cliente para probar la API
└── README.md               # Documentación
