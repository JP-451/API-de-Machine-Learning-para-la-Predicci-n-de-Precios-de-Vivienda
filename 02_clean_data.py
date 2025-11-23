import pandas as pd
import numpy as np
import sys

input_file = "data/combined_housing_data.csv"
output_file = "data/cleaned_housing_data.csv"

try:
    df = pd.read_csv(input_file)
    print(f"Archivo '{input_file}' cargando. {len(df)} filas iniciales.")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{input_file}'.")
    print(f"Asegurate que el archivo esté en la misma carpeta que el archivo.")
    sys.exit()

except pd.errors.EmptyDataError:
    print(f"Error: El archivo '{input_file}' está vacío.")
    sys.exit()

columns_to_keep = [
    'price', 'm2_real', 'room_num', 'bath_num', 'province', 'house_type', 'condition', 'lift', 
    'balcony', 'terrace', 'swimming_pool', 'garden', 'garage', 'built_in_wardrobe', 'air_conditioner'
] 

available_columns = [col for col in columns_to_keep if col in df.columns]
print(f"Columnas seleccionadas: {available_columns}")
df = df[available_columns]

print("Convertir columnas 'price', 'm2_real', 'room_num', 'bath_num' a números...")
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['m2_real'] = pd.to_numeric(df['m2_real'], errors='coerce')
df['room_num'] = pd.to_numeric(df['room_num'], errors='coerce')
df['bath_num'] = pd.to_numeric(df['bath_num'], errors='coerce')

df.dropna(subset=['price', 'm2_real'], inplace=True)
print(f"Filas despues de eliminar las que tienen 'price' o 'm2_real' nulos: {len(df)}")

df['room_num'].fillna(0, inplace=True)
df['bath_num'].fillna(0, inplace=True)

df['room_num'] = df['room_num'].astype(int)
df['bath_num'] = df['bath_num'].astype(int)

print("Procesando las columnas restantes (lift, terrace, ect...)")
boolean_cols = [
    'lift', 'balcony', 'terrace', 'swimming_pool', 'garden', 
    'garage', 'built_in_wardrobe', 'air_conditioner'  
]

for col in boolean_cols:
    if col in df.columns:

        df[col] = df[col].astype(str).str.lower()

        df[col] = np.where(df[col].isin(['0', 'nan']), 0, 1)
        df[col] = df[col].astype(int)

if 'condition' in df.columns:
    df['condition'].fillna('Unknown', inplace=True)

if 'house_type' in df.columns:
    df['house_type'].fillna('Unknown', inplace=True)

df.to_csv(output_file, index=False)

print(f"\n¡Éxito! Archivo limpiado guardado como '{output_file}'.")
print(f"Total de filas limpias: {len(df)}")
print("\n--- Información del DataFrame Limpio ---")
df.info()