import pandas as pd
import glob
import os
import sys

file_pattern = "data/houses_*.csv"
output_file = "data/combined_housing_data.csv"

file_list = glob.glob(file_pattern)

if not file_list:
    print(f"Error: No se encontraron archivos con el patrón '{file_pattern}'.")
    print("Asegúrate de que tus CSV están en la misma carpeta que este script.")
    sys.exit()
else:
    print(f"¡Éxito! Se encontraron {len(file_list)} archivos para combinar:")
    print(file_list)
    
    all_dataframes = []
    
    for filename in file_list:
        try:
            df = pd.read_csv(filename)
            province_name = os.path.basename(filename).replace('houses_', '').replace('.csv', '').replace('_', ' ').title()
            df['province'] = province_name
            all_dataframes.append(df)

        except pd.errors.EmptyDataError:
            print(f"Advertencia: El archivo {filename} está vacío y será omitido.")

        except Exception as e:
            print(f"Error procesando el archivo {filename}: {e}")
    
    if all_dataframes:
        master_df = pd.concat(all_dataframes, ignore_index=True)
        master_df.to_csv(output_file, index=False)
        print(f"\n¡Perfecto! Archivo combinado guardado como '{output_file}'.")
        print(f"Total de filas: {len(master_df)}")

    else:
        print("Error: No se pudo crear el archivo combinado.")
        sys.exit()