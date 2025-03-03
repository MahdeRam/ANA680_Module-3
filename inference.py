import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
with open('wine_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['fixed_acidity'], data['volatile_acidity'], data['citric_acid'],
                data['residual_sugar'], data['chlorides'], data['free_sulfur_dioxide'],
                data['total_sulfur_dioxide'], data['density'], data['pH'],
                data['sulphates'], data['alcohol']]
    
    prediction = model.predict([features])
    return jsonify({'quality_prediction': float(prediction[0])})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)