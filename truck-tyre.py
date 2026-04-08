from pathlib import Path
import os
import logging

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "tire_test_data_large.csv"
PRESSURE_FILE = BASE_DIR / "fleet_tire_dataset_real.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {BASE_DIR}")
    df = pd.read_csv(path)
    print(f"✅ {path.name} uploaded to system for analysis")
    return df


def normalize_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def safe_int(value, field_name: str) -> int:
    try:
        return int(value)
    except Exception:
        raise ValueError(f"'{field_name}' must be an integer")


def safe_float(value, field_name: str) -> float:
    try:
        return float(value)
    except Exception:
        raise ValueError(f"'{field_name}' must be a number")


def fit_encoder(values: pd.Series) -> LabelEncoder:
    enc = LabelEncoder()
    enc.fit(sorted(values.dropna().astype(str).str.strip().str.lower().unique()))
    return enc


data = load_csv(TRAIN_FILE)
pressure_data = load_csv(PRESSURE_FILE)

required_train_cols = {"roadtype", "loadkg", "axles", "climate", "steertire", "drivetire", "trailertire"}
required_pressure_cols = {
    "roadtype", "loadkg", "axles", "climate",
    "temperature", "avg_speed", "tire_age", "wear_level", "optimal_psi"
}

missing_train = required_train_cols - set(data.columns)
missing_pressure = required_pressure_cols - set(pressure_data.columns)

if missing_train:
    raise ValueError(f"tire_test_data_large.csv missing columns: {sorted(missing_train)}")

if missing_pressure:
    raise ValueError(f"fleet_tire_dataset_real.csv missing columns: {sorted(missing_pressure)}")

data["roadtype"] = normalize_series(data["roadtype"])
data["climate"] = normalize_series(data["climate"])
data["steertire"] = normalize_series(data["steertire"])
data["drivetire"] = normalize_series(data["drivetire"])
data["trailertire"] = normalize_series(data["trailertire"])

pressure_data["roadtype"] = normalize_series(pressure_data["roadtype"])
pressure_data["climate"] = normalize_series(pressure_data["climate"])
pressure_data["wear_level"] = normalize_series(pressure_data["wear_level"])

road_encoder = fit_encoder(pd.concat([data["roadtype"], pressure_data["roadtype"]], ignore_index=True))
climate_encoder = fit_encoder(pd.concat([data["climate"], pressure_data["climate"]], ignore_index=True))
wear_encoder = fit_encoder(pressure_data["wear_level"])

steer_encoder = fit_encoder(data["steertire"])
drive_encoder = fit_encoder(data["drivetire"])
trailer_encoder = fit_encoder(data["trailertire"])

data["road_enc"] = road_encoder.transform(data["roadtype"])
data["climate_enc"] = climate_encoder.transform(data["climate"])

pressure_data["road_enc"] = road_encoder.transform(pressure_data["roadtype"])
pressure_data["climate_enc"] = climate_encoder.transform(pressure_data["climate"])
pressure_data["wear_enc"] = wear_encoder.transform(pressure_data["wear_level"])

data["steer_enc"] = steer_encoder.transform(data["steertire"])
data["drive_enc"] = drive_encoder.transform(data["drivetire"])
data["trailer_enc"] = trailer_encoder.transform(data["trailertire"])

X = data[["road_enc", "loadkg", "axles", "climate_enc"]]

steer_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
drive_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
trailer_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

steer_model.fit(X, data["steer_enc"])
drive_model.fit(X, data["drive_enc"])
trailer_model.fit(X, data["trailer_enc"])

pressure_data["load_per_axle"] = pressure_data["loadkg"] / pressure_data["axles"]

pressure_features = [
    "road_enc",
    "loadkg",
    "axles",
    "climate_enc",
    "temperature",
    "avg_speed",
    "tire_age",
    "wear_enc",
    "load_per_axle",
]

pressure_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
pressure_model.fit(pressure_data[pressure_features], pressure_data["optimal_psi"])

app = Flask(__name__)
CORS(app)
app.config["JSON_SORT_KEYS"] = False


@app.route("/")
def home():
    return jsonify({
        "message": "Truck Tire Optimization API Running",
        "endpoints": ["/predict"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.get_json(silent=True) or {}

        required_fields = [
            "roadType", "loadKg", "axles", "climate",
            "temperature", "speed", "tireAge", "wearLevel"
        ]
        missing = [f for f in required_fields if f not in req]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        roadtype = str(req["roadType"]).strip().lower()
        loadkg = safe_int(req["loadKg"], "loadKg")
        axles = safe_int(req["axles"], "axles")
        climate = str(req["climate"]).strip().lower()
        temperature = safe_float(req["temperature"], "temperature")
        speed = safe_float(req["speed"], "speed")
        tire_age = safe_float(req["tireAge"], "tireAge")
        wear = str(req["wearLevel"]).strip().lower()

        if loadkg <= 0:
            return jsonify({"error": "loadKg must be greater than 0"}), 400
        if axles <= 0:
            return jsonify({"error": "axles must be greater than 0"}), 400

        if roadtype not in road_encoder.classes_:
            return jsonify({"error": f"Invalid roadType: {roadtype}"}), 400
        if climate not in climate_encoder.classes_:
            return jsonify({"error": f"Invalid climate: {climate}"}), 400
        if wear not in wear_encoder.classes_:
            return jsonify({"error": f"Invalid wearLevel: {wear}"}), 400

        road_enc = road_encoder.transform([roadtype])[0]
        climate_enc = climate_encoder.transform([climate])[0]
        wear_enc = wear_encoder.transform([wear])[0]

        load_per_axle = loadkg / axles

        X_input = pd.DataFrame([{
            "road_enc": road_enc,
            "loadkg": loadkg,
            "axles": axles,
            "climate_enc": climate_enc
        }])

        steer_idx = steer_model.predict(X_input)[0]
        drive_idx = drive_model.predict(X_input)[0]
        trailer_idx = trailer_model.predict(X_input)[0]

        steer = steer_encoder.inverse_transform([steer_idx])[0]
        drive = drive_encoder.inverse_transform([drive_idx])[0]
        trailer = trailer_encoder.inverse_transform([trailer_idx])[0]

        pressure_input = pd.DataFrame([{
            "road_enc": road_enc,
            "loadkg": loadkg,
            "axles": axles,
            "climate_enc": climate_enc,
            "temperature": temperature,
            "avg_speed": speed,
            "tire_age": tire_age,
            "wear_enc": wear_enc,
            "load_per_axle": load_per_axle
        }])

        pressure = float(pressure_model.predict(pressure_input)[0])

        risk_score = 0

        if load_per_axle > 4000:
            risk_score += 2
        elif load_per_axle > 2500:
            risk_score += 1

        if speed > 90:
            risk_score += 2
        elif speed > 70:
            risk_score += 1

        if temperature > 40:
            risk_score += 1

        if tire_age > 4:
            risk_score += 2
        elif tire_age > 2:
            risk_score += 1

        if wear == "high":
            risk_score += 2
        elif wear == "medium":
            risk_score += 1

        if risk_score >= 6:
            risk = "HIGH"
            advice = "Replace tire immediately"
        elif risk_score >= 3:
            risk = "MEDIUM"
            advice = "Check tire soon"
        else:
            risk = "LOW"
            advice = "Tire condition is safe"

        return jsonify({
            "steerTire": steer,
            "driveTire": drive,
            "trailerTire": trailer,
            "pressure": round(pressure, 2),
            "failureRisk": risk,
            "loadPerAxle": round(load_per_axle, 2),
            "safetyAdvice": advice
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=True)