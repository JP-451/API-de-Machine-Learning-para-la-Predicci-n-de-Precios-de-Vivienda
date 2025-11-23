import pandas as pd
import joblib
import json
from flask import Flask, request, jsonify 
import sys

app = Flask(__name__)

model_file = "data/housing_price_model.pkl"
columns_file = "data/model_columns.json"

try:
    model = joblib.load(model_file)
    print(f"Modelo '{model_file}' cargado exitosamente")
except FileNotFoundError:
    print(f"Error: Archivo del modelo '{model_file}' no encontrado")
    sys.exit()

try:
    with open(columns_file, 'r') as f:
        model_columns = json.load(f)
    print(f"Columnas del modelo '{columns_file}' cargadas exitosamente.")
except FileNotFoundError:
    print(f"Error: Archivo de columnas '{columns_file}' no enconrtado")
    sys.exit()

def prepare_data(input_json, columns_list):
    data_row = {col: 0 for col in columns_list}

    for key, value in input_json.items():
        if key in data_row:
            data_row[key] = value
    
    if 'province' in input_json:
        col_name = f"province_{input_json['province']}"
        if col_name in data_row:
            data_row[col_name] = 1
    
    if 'house_type' in input_json:
        col_name = f"house_type_{input_json['house_type']}"
        if col_name in data_row:
            data_row[col_name] = 1

    if 'condition' in input_json:
        col_name = f"condition_{input_json['condition']}"
        if col_name in data_row:
            data_row[col_name] = 1
    
    df_row = pd.DataFrame(data_row, index=[0])
    df_row = df_row[columns_list]

    return df_row

@app.route("/predict", methods=['POST'])
def predict():
    input_json = request.json
    try:
        processed_data = prepare_data(input_json, model_columns)

        prediction = model.predict(processed_data)

        return jsonify({
            "precio_estimado": round(prediction[0], 2),
            "input_data": input_json
        })
    except Exception as e:
        return jsonify({"Error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)