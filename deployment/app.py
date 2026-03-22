from flask import Flask, render_template, request
import numpy as np
import pickle
feature_order = pickle.load(open("columns.pkl","rb"))

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("best_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# IMPORTANT: same order as training data
feature_order = [
    'A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
    'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score',
    'age','gender','ethnicity','jaundice','austim',
    'contry_of_res','used_app_before','relation'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.form.to_dict()
        processed_data = []

        for col in feature_order:
            value = input_data[col]

            # Encode categorical
            if col in encoders:
                value = encoders[col].transform([value])[0]

            processed_data.append(float(value))

        final_input = np.array([processed_data])

        prediction = model.predict(final_input)[0]

        result = "Autism Detected " if prediction == 1 else "No Autism "

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {e}"

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)