import requests
import json

datos_casa = {
    "m2_real": 100,
    "room_num": 3,
    "bath_num": 2,
    "lift": 1,
    "garage": 1,
    "terrace": 1,
    "province": "Madrid",
    "house_type": "Piso",
    "condition": "segunda mano/buen estado"
}

url = "http://127.0.0.1:5000/predict"

print(f"Enviando datos a la API en {url}...")
print(f"Datos: {json.dumps(datos_casa, indent=2)}")

try:
    response = requests.post(url, json=datos_casa)
    
    if response.status_code == 200:
        # ¡ÉXITO!
        print("\n--- ¡Predicción Recibida! ---")
        print(response.json())
    else:
        print(f"\n--- Error: El servidor respondió con el código {response.status_code} ---")
        print(response.text) # Imprimir el error que el servidor nos dio

except requests.exceptions.ConnectionError:
    print("\n--- ERROR DE CONEXIÓN ---")
    print("No se pudo conectar a la API.")
    print("¿Estás seguro de que 'app.py' se está ejecutando en la otra terminal?")
except Exception as e:
    print(f"\nHa ocurrido un error inesperado: {e}")