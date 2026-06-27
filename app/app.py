from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = []

        for feature in FEATURE_NAMES:
            value = request.form.get(feature)

            if value is None or value == "":
                return render_template(
                    "index.html",
                    prediction_text="Erreur : veuillez remplir tous les champs."
                )

            features.append(float(value))

        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability_text = ""

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0][int(prediction)]
            probability_text = f"Probabilité estimée : {probability * 100:.2f} %"

        if prediction == 1:
            result = "Risque de maladie cardiaque détecté"
        else:
            result = "Aucun risque de maladie cardiaque détecté"

        return render_template(
            "index.html",
            prediction_text=result,
            probability_text=probability_text
        )

    except ValueError:
        return render_template(
            "index.html",
            prediction_text="Erreur : veuillez saisir uniquement des valeurs numériques."
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Erreur lors de la prédiction : {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
