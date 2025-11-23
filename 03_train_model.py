import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import json
import sys

input_file = "data/cleaned_housing_data.csv"
output_model_file = "data/housing_price_model.pkl"
output_columns_file = "data/model_columns.json"

try:
    df = pd.read_csv(input_file)
    print(f"Archivo '{input_file}' cargando. {len(df)} filas.")

except FileNotFoundError:
    print(f"Error: No se ha encontrado el archivo '{input_file}'.")
    print("Asegurate de que se ha ejecuta el archivo '02_clean_data.py' primero.")
    sys.exit()

except pd.errors.EmptyDataError:
    print(f"Error: El archivo '{input_file}' está vacío.")
    sys.exit()

q_99_m2 = df['m2_real'].quantile(0.99)
q_99_price = df['price'].quantile(0.99)

print(f"Filtrando outliers: m2_real <= {q_99_m2}, price <= {q_99_price}, room <= 10, bath_room <= 6")

df_processed = df[
    (df["m2_real"] <= q_99_m2) &
    (df["price"] <= q_99_price) &
    (df["room_num"] <= 10) &
    (df["bath_num"] <= 6) &
    (df["m2_real"] >= 15)
]

print(f"Filas después de eleminiar outliers: {len(df_processed)}. (Eliminadas: {len(df) - len(df_processed)})")

print("Codificando variables categóricas (province, house_type, condition)")
df_processed = pd.get_dummies(df_processed, columns=['province', 'house_type', 'condition'], drop_first=True)

y = df_processed['price']
X = df_processed.drop('price', axis=1)

model_columns = X.columns.to_list()
with open(output_columns_file, 'w') as f:
    json.dump(model_columns, f)
print(f"Lista de {len(model_columns)} columnas del modelo guardada en '{output_columns_file}'.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Datos divididos: {len(X_train)} filas para entrenar, {len(X_test)} filas para probar.")

print("Entrenando el modelo RandomForestRegressor... (Esto puede tardar unos segundos)")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1, oob_score=True)
model.fit(X_train, y_train)

# Puntuación Out-of-Bag (OOB) es una buena métrica de rendimiento
oob_score = model.oob_score_
print(f"\n--- Evaluación del Modelo Completada!!!")
print(f"Puntuacion Out-Of-Bag (R²):     {oob_score:.4f}")

# Evaluación en el set de prueba (Test) para confirmar
y_pred = model.predict(X_test)
test_score = r2_score(y_test, y_pred)
print(f"Puntuacion en Test (R²):     {test_score:.4f}")

print("\n(Una puntuación R² cercana a 1.0 es ideal. > 0.6 es buena para este tipo de datos)")

joblib.dump(model, output_model_file)
print(f"\n¡Exito! Modelo guardado como '{output_model_file}'.")
