import joblib
import os
import numpy as np

def model_fn(model_dir):
    """Load the trained model from the SageMaker model directory"""
    model_path = os.path.join(model_dir, "model.joblib")
    print(f"Loading model from {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Deserialize JSON input data"""
    if request_content_type == "application/json":
        import json
        data = json.loads(request_body)
        return np.array(data["instances"])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction using the loaded model"""
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, response_content_type):
    """Serialize the output as JSON"""
    if response_content_type == "application/json":
        import json
        return json.dumps({"predictions": prediction.tolist()})
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
