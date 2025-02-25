import numpy as np
import joblib
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load trained model
try:
    model = joblib.load("wine_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Extract input features from the form
            features = [
                float(request.form["fixed_acidity"]),
                float(request.form["volatile_acidity"]),
                float(request.form["citric_acid"]),
                float(request.form["residual_sugar"]),
                float(request.form["chlorides"]),
                float(request.form["free_sulfur_dioxide"]),
                float(request.form["total_sulfur_dioxide"]),
                float(request.form["density"]),
                float(request.form["pH"]),
                float(request.form["sulphates"]),
                float(request.form["alcohol"]),
            ]

            # Convert to NumPy array and reshape for prediction
            features_array = np.array(features).reshape(1, -1)

            # Make prediction
            prediction = round(model.predict(features_array)[0], 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("wine_form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)





