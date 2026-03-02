import numpy as np
import pandas as pd
import ipaddress
from flask import Flask, request, jsonify
import io
import joblib

app = Flask(__name__)
def ip_to_int(ip):
    return int(ipaddress.ip_address(ip))

def load_and_preprocess_data_ml(data):

    try:
        data['Login_Hour'] = pd.to_datetime(data['Login_Timestamp']).dt.hour
        data['Login_Successful'] = data['Login_Successful'].astype(np.uint8)
        data = data.drop(columns=["Round_Trip_Time", 'Region', 'City', 'Login_Timestamp', 'index'])

        data['User_Agent_String'], _ = pd.factorize(data['User_Agent_String'])
        data['Browser_Name_and_Version'], _ = pd.factorize(data['Browser_Name_and_Version'])
        data['OS_Name_and_Version'], _ = pd.factorize(data['OS_Name_and_Version'])
        data['IP_Address'] = data['IP_Address'].apply(ip_to_int)

        data['Is_Unknown_Country'] = 0

        pipeline = joblib.load(f'data/Models/{data.iloc[0]["User_ID"]}.pkl')
        predictions = pipeline.predict_proba(data)[:, 1]

        return predictions
    except Exception as e:
        print("Ошибка при обработке JSON:", e)
        return jsonify({"error": str(e)}), 400


@app.route('/', methods=['POST'])
def handle_request():
    json_data = f'[{request.data.decode("utf-8")}]'
    try:
        data = pd.read_json(io.StringIO(json_data))
        predictions = float(load_and_preprocess_data_ml(data))

        with open(f'data/LevelRisk/{data.iloc[0]["User_ID"]}.txt', 'r') as f:
            lines = f.readlines()
            level_risk = {
                'low_medium': float(lines[0].strip()),
                'medium_high': float(lines[1].strip())
            }


        if(predictions <= level_risk['low_medium']):
            risk_analysis = 'min'
        elif (predictions > level_risk['low_medium'] and predictions < level_risk['medium_high']):
            risk_analysis = 'middle'
        else:
            risk_analysis = 'high'

        response_data = pd.DataFrame({'probability': [risk_analysis]})
        response_json = response_data.to_json(orient='records', index=False)[1:-1]

        return response_json, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        print("Ошибка при обработке CSV:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9292)
