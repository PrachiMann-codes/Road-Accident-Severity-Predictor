from flask import Flask, render_template, request
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
import pandas as pd
import webbrowser

app = Flask(__name__)

# Load preprocessor and XGB model
preprocessor = joblib.load("preprocessor.joblib")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_model.json")

# Rebuild pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Collect inputs in the same order you trained
        features = {
            "Number_of_vehicles_involved": int(request.form.get("Number_of_vehicles_involved")),
            "Number_of_casualties": int(request.form.get("Number_of_casualties")),
            "Types_of_Junction": request.form.get("Types_of_Junction"),
            "Light_conditions": request.form.get("Light_conditions"),
            "Weather_conditions": request.form.get("Weather_conditions"),
            "Hour": int(request.form.get("Hour")),
            "Day_of_week": request.form.get("Day_of_week"),
            "Age_band_of_driver": request.form.get("Age_band_of_driver"),
            "Pedestrian_movement": request.form.get("Pedestrian_movement"),
            "Type_of_collision": request.form.get("Type_of_collision")
        }

       # Convert to DataFrame (single row)
        input_df = pd.DataFrame([features])

        # Predict
        prediction = model.predict(input_df)[0]

        severity_map = {0: "Slight Injury", 1: "Serious Injury", 2: "Fatal Injury"}
        result = severity_map[prediction]

        return render_template("predict.html", 
                       prediction_text=f"Predicted Severity: {result}",
                       form_values=features)


    return render_template("predict.html", prediction_text=None)

if __name__ == "__main__":
    webbrowser.get("windows-default").open("http://127.0.0.1:5000/")
    app.run(debug=True)
