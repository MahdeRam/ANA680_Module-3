from flask import Flask, request, render_template
import joblib
import numpy as np

# Load trained model
model = joblib.load("breast_cancer_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)  # Initialize with no prediction

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract all 9 features from form
        features = [
            float(request.form['clump_thickness']),
            float(request.form['uniformity_cell_size']),
            float(request.form['uniformity_cell_shape']),
            float(request.form['marginal_adhesion']),
            float(request.form['single_epithelial_cell_size']),
            float(request.form['bare_nuclei']),
            float(request.form['bland_chromatin']),
            float(request.form['normal_nucleoli']),
            float(request.form['mitoses'])
        ]

        # Convert to numpy array for prediction
        input_data = np.array([features])  # Ensure it's a 2D array
        prediction = model.predict(input_data)[0]

        # Convert result to readable format
        predicted_class = "Malignant" if prediction == 1 else "Benign"

        return render_template('index.html', prediction=predicted_class)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)



