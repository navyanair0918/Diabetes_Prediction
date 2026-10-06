"""DiaLens: demonstration interface for the BRFSS diabetes-status classifier."""
import json
import pickle
from io import BytesIO
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from ml_pipeline import FEATURES, LABELS, MODEL_PATH, REPORT_PATH

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

BINARY_OPTIONS = [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]
OPTIONAL_FIELDS = {"Sex", "Age", "Education", "Income"}
FIELDS = [
    {"name": "HighBP", "label": "High blood pressure", "group": "Health history", "type": "select", "options": BINARY_OPTIONS, "help": "Ever told you have high blood pressure or are taking medicine for it."},
    {"name": "HighChol", "label": "High cholesterol", "group": "Health history", "type": "select", "options": BINARY_OPTIONS, "help": "Ever told you have high blood cholesterol."},
    {"name": "CholCheck", "label": "Cholesterol check in past 5 years", "group": "Health history", "type": "select", "options": BINARY_OPTIONS, "help": "Whether cholesterol was checked within the survey's five-year window."},
    {"name": "Stroke", "label": "History of stroke", "group": "Health history", "type": "select", "options": BINARY_OPTIONS, "help": "Self-reported history."},
    {"name": "HeartDiseaseorAttack", "label": "Heart disease or heart attack", "group": "Health history", "type": "select", "options": BINARY_OPTIONS, "help": "Self-reported coronary heart disease or heart attack."},
    {"name": "DiffWalk", "label": "Difficulty walking or climbing stairs", "group": "Health history", "type": "select", "options": BINARY_OPTIONS, "help": "Self-reported difficulty due to a health problem."},
    {"name": "BMI", "label": "Body mass index (BMI)", "group": "Lifestyle", "type": "number", "min": 12, "max": 98, "step": 0.1, "value": 27, "unit": "kg/m²", "help": "Use a value in the range represented by the source survey."},
    {"name": "Smoker", "label": "Smoked at least 100 cigarettes in life", "group": "Lifestyle", "type": "select", "options": BINARY_OPTIONS, "help": "Survey indicator; this is not a current-smoking measure."},
    {"name": "PhysActivity", "label": "Physical activity in past 30 days", "group": "Lifestyle", "type": "select", "options": BINARY_OPTIONS, "help": "Any physical activity outside regular job activity."},
    {"name": "Fruits", "label": "Fruit consumption indicator", "group": "Lifestyle", "type": "select", "options": BINARY_OPTIONS, "help": "Survey-coded daily fruit consumption indicator."},
    {"name": "Veggies", "label": "Vegetable consumption indicator", "group": "Lifestyle", "type": "select", "options": BINARY_OPTIONS, "help": "Survey-coded daily vegetable consumption indicator."},
    {"name": "HvyAlcoholConsump", "label": "Heavy alcohol consumption indicator", "group": "Lifestyle", "type": "select", "options": BINARY_OPTIONS, "help": "The survey's derived indicator; not a personalized assessment."},
    {"name": "GenHlth", "label": "General health", "group": "Health status", "type": "select", "options": [{"value": 1, "label": "Excellent"}, {"value": 2, "label": "Very good"}, {"value": 3, "label": "Good"}, {"value": 4, "label": "Fair"}, {"value": 5, "label": "Poor"}], "help": "Ordered survey response from excellent (1) to poor (5)."},
    {"name": "MentHlth", "label": "Days mental health was not good", "group": "Health status", "type": "number", "min": 0, "max": 30, "step": 1, "value": 0, "unit": "days in past 30", "help": "Survey response from 0 to 30 days."},
    {"name": "PhysHlth", "label": "Days physical health was not good", "group": "Health status", "type": "number", "min": 0, "max": 30, "step": 1, "value": 0, "unit": "days in past 30", "help": "Survey response from 0 to 30 days."},
    {"name": "AnyHealthcare", "label": "Any healthcare coverage", "group": "Access & demographics", "type": "select", "options": BINARY_OPTIONS, "help": "Self-reported coverage indicator."},
    {"name": "NoDocbcCost", "label": "Could not see a doctor because of cost", "group": "Access & demographics", "type": "select", "options": BINARY_OPTIONS, "help": "Self-reported access barrier in the past 12 months."},
    {"name": "Sex", "label": "Sex category in source survey", "group": "Access & demographics", "type": "select", "options": [{"value": 0, "label": "Female (survey code 0)"}, {"value": 1, "label": "Male (survey code 1)"}], "help": "Retains the source data's binary coding."},
    {"name": "Age", "label": "Age group", "group": "Access & demographics", "type": "select", "options": [{"value": 1, "label": "18–24"}, {"value": 2, "label": "25–29"}, {"value": 3, "label": "30–34"}, {"value": 4, "label": "35–39"}, {"value": 5, "label": "40–44"}, {"value": 6, "label": "45–49"}, {"value": 7, "label": "50–54"}, {"value": 8, "label": "55–59"}, {"value": 9, "label": "60–64"}, {"value": 10, "label": "65–69"}, {"value": 11, "label": "70–74"}, {"value": 12, "label": "75–79"}, {"value": 13, "label": "80+"}], "help": "Ordinal age-group code, not exact age."},
    {"name": "Education", "label": "Education level", "group": "Access & demographics", "type": "select", "options": [{"value": 1, "label": "Never attended / kindergarten"}, {"value": 2, "label": "Grades 1–8"}, {"value": 3, "label": "Grades 9–11"}, {"value": 4, "label": "High school graduate / GED"}, {"value": 5, "label": "College 1–3 years"}, {"value": 6, "label": "College 4+ years"}], "help": "Ordinal survey category."},
    {"name": "Income", "label": "Annual household income band", "group": "Access & demographics", "type": "select", "options": [{"value": 1, "label": "Under $10,000"}, {"value": 2, "label": "$10,000–$14,999"}, {"value": 3, "label": "$15,000–$19,999"}, {"value": 4, "label": "$20,000–$24,999"}, {"value": 5, "label": "$25,000–$34,999"}, {"value": 6, "label": "$35,000–$49,999"}, {"value": 7, "label": "$50,000–$74,999"}, {"value": 8, "label": "$75,000+"}], "help": "Ordinal 2015 survey category; income groups may not transfer to other places or years."},
]

if not MODEL_PATH.exists() or not REPORT_PATH.exists():
    raise RuntimeError("Model files are missing. Run 'python train_model.py' in the project folder first.")
with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)
with REPORT_PATH.open(encoding="utf-8") as report_file:
    report = json.load(report_file)


@app.get("/")
def home():
    groups = {}
    for field in FIELDS:
        groups.setdefault(field["group"], []).append(field)
    return render_template("home.html", report=report, fields=FIELDS,
                           field_groups=groups, labels=LABELS,
                           optional_fields=OPTIONAL_FIELDS)


_dataset_defaults = None


def get_dataset_defaults():
    """Use survey-typical values only when someone opts out of demographics."""
    global _dataset_defaults
    if _dataset_defaults is None:
        if not DATA_PATH.exists():
            raise ValueError("The survey dataset is needed when using 'Prefer not to say'.")
        _dataset_defaults = pd.read_csv(DATA_PATH, usecols=FEATURES).mode().iloc[0]
    return _dataset_defaults


def parse_form(form):
    row = {}
    omitted = []
    try:
        height_cm = float(form["height"])
        weight_kg = float(form["weight"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Enter a valid height and weight to calculate BMI.")
    if not 100 <= height_cm <= 250 or not 25 <= weight_kg <= 300:
        raise ValueError("Height must be 100–250 cm and weight must be 25–300 kg.")
    calculated_bmi = weight_kg / ((height_cm / 100) ** 2)
    if not 12 <= calculated_bmi <= 98:
        raise ValueError("Calculated BMI must be between 12 and 98 for this model.")
    for field in FIELDS:
        name = field["name"]
        if name == "BMI":
            value = calculated_bmi
        elif form.get(name) == "prefer_not_to_say" and name in OPTIONAL_FIELDS:
            value = float(get_dataset_defaults()[name])
            omitted.append(field["label"])
        else:
            try:
                value = float(form[name])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Enter a valid value for {field['label']}.")
        if field["type"] == "select":
            allowed = {float(option["value"]) for option in field["options"]}
            if value not in allowed:
                raise ValueError(f"Choose a listed value for {field['label']}.")
        elif not field["min"] <= value <= field["max"]:
            raise ValueError(f"{field['label']} must be between {field['min']} and {field['max']}.")
        row[name] = value
    return pd.DataFrame([row], columns=FEATURES), omitted


def bmi_category(bmi):
    if bmi < 18.5:
        return "below the typical adult range"
    if bmi < 25:
        return "within the typical adult range"
    if bmi < 30:
        return "in the higher adult range"
    return "in the obesity range for adults"


@app.post("/predict")
def predict():
    try:
        frame, omitted = parse_form(request.form)
        score = float(model.predict_proba(frame)[0, 1])
        predicted = int(score >= report["threshold"])
        answers = frame.iloc[0]
        bmi = float(answers["BMI"])
        profile_details = [
            f"Calculated BMI: {bmi:.1f} ({bmi_category(bmi)}).",
            f"High blood pressure: {'Yes' if answers['HighBP'] else 'No'}.",
            f"High cholesterol: {'Yes' if answers['HighChol'] else 'No'}.",
            f"Physical activity in the past 30 days: {'Yes' if answers['PhysActivity'] else 'No'}.",
        ]
        if omitted:
            profile_details.append(
                "Prefer-not-to-say fields use the most common value in the source dataset: "
                + ", ".join(omitted) + "."
            )
        discussion_points = []
        if predicted:
            discussion_points.append("Ask a healthcare professional whether a blood glucose test is appropriate.")
        if answers["HighBP"]:
            discussion_points.append("Discuss blood pressure follow-up and how to manage it safely.")
        if answers["HighChol"]:
            discussion_points.append("Discuss cholesterol screening and any recommended follow-up.")
        if answers["PhysActivity"] == 0:
            discussion_points.append("Ask about a safe, manageable way to add regular activity.")
        if bmi >= 25:
            discussion_points.append("Ask whether any weight-related goals are appropriate for you; BMI is only a rough screening measure.")
        if not discussion_points:
            discussion_points.append("Continue routine checkups and ask about testing if you have symptoms or concerns.")
        return jsonify(
            predicted_class=predicted,
            label=LABELS[predicted],
            score=round(score, 4),
            threshold=report["threshold"],
            other_score=round(1 - score, 4),
            bmi=round(bmi, 1),
            bmi_category=bmi_category(bmi),
            profile_details=profile_details,
            discussion_points=discussion_points,
        )
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.post("/batch")
def batch_predict():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename.lower().endswith(".csv"):
        return jsonify(error="Choose a CSV file."), 400
    try:
        batch = pd.read_csv(uploaded.stream)
        missing = [name for name in FEATURES if name not in batch.columns]
        if missing:
            return jsonify(error="CSV is missing columns: " + ", ".join(missing)), 400
        frame = batch[FEATURES].apply(pd.to_numeric, errors="raise")
        if frame.empty:
            return jsonify(error="CSV contains no rows."), 400
        scores = model.predict_proba(frame)[:, 1]
        batch["ModelScore_PositiveClass"] = scores.round(4)
        batch["PredictedClass"] = (scores >= report["threshold"]).astype(int)
        output = BytesIO()
        batch.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype="text/csv", as_attachment=True,
                         download_name="dialens_scored_results.csv")
    except Exception:
        return jsonify(error="Could not score this file. Check that it contains numeric values and all required feature columns."), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0" if os.environ.get("CODESPACES") == "true" else "127.0.0.1", port=5000, debug=False)
