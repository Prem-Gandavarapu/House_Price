from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

MODEL_DIR = "models"

models = {
    "Linear Regression": joblib.load(os.path.join(MODEL_DIR, "linear_regression.pkl")),
    "Ridge": joblib.load(os.path.join(MODEL_DIR, "ridge.pkl")),
    "Lasso": joblib.load(os.path.join(MODEL_DIR, "lasso.pkl")),
    "Elastic Net": joblib.load(os.path.join(MODEL_DIR, "elastic_net.pkl")),
    "Polynomial Regression": joblib.load(
        os.path.join(MODEL_DIR, "polynomial_regression.pkl")
    )
}

features = joblib.load(
    os.path.join(MODEL_DIR, "features.pkl")
)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "House Price Prediction API",
        "models": list(models.keys())
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict house price using all trained models
    ---
    tags:
      - Prediction
    parameters:
      - in: body
        name: house
        required: true
        schema:
          type: object
    responses:
      200:
        description: Predicted house prices
    """

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        input_data = pd.DataFrame([data])
        input_data = input_data.reindex(columns=features)

        predictions = {}

        for name, model in models.items():
            predictions[name] = float(model.predict(input_data)[0])

        return jsonify(predictions)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400



if __name__ == "__main__":
    app.run(debug=True)