from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os


# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="Weather Rain Prediction API",
    description="API for predicting rain using ML models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Model Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

RANDOM_FOREST_PATH = os.path.join(
    MODEL_DIR,
    "random_forest.joblib"
)

DECISION_TREE_PATH = os.path.join(
    MODEL_DIR,
    "decision_tree.joblib"
)

XGBOOST_PATH = os.path.join(
    MODEL_DIR,
    "xgboost.joblib"
)


# ==========================================
# Load Models
# ==========================================

try:
    random_forest_model = joblib.load(RANDOM_FOREST_PATH)
    decision_tree_model = joblib.load(DECISION_TREE_PATH)
    xgboost_model = joblib.load(XGBOOST_PATH)

    models_loaded = True

except Exception as e:
    print("Model loading error:", e)
    models_loaded = False


# ==========================================
# Input Schema
# ==========================================

class WeatherInput(BaseModel):
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Cloud_Cover: float
    Pressure: float


# ==========================================
# Input + Model Schema
# ==========================================

class WeatherModelInput(BaseModel):
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Cloud_Cover: float
    Pressure: float
    model: str


# ==========================================
# Helper Function
# ==========================================

def get_features(data):
    return [[
        data.Temperature,
        data.Humidity,
        data.Wind_Speed,
        data.Cloud_Cover,
        data.Pressure
    ]]


def format_prediction(model, features):

    prediction = model.predict(features)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probability = float(
            model.predict_proba(features)[0][1]
        )

    if prediction == 1:
        result = "rain"
    else:
        result = "no rain"

    response = {
        "prediction": result,
        "prediction_code": int(prediction)
    }

    if probability is not None:
        response["rain_probability"] = round(probability, 4)

    return response


# ==========================================
# API 1: Health Check
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "models": [
            "Random Forest",
            "Decision Tree",
            "XGBoost"
        ]
    }


# ==========================================
# API 2: Random Forest Prediction
# ==========================================

@app.post("/predict")
def predict_rain(data: WeatherInput):

    if not models_loaded:
        raise HTTPException(
            status_code=500,
            detail="Models are not loaded"
        )

    features = get_features(data)

    result = format_prediction(
        random_forest_model,
        features
    )

    return {
        "model": "Random Forest",
        "input": data.model_dump(),
        **result
    }


# ==========================================
# API 3: User Selects Model
# ==========================================

@app.post("/predict/model")
def predict_with_selected_model(data: WeatherModelInput):

    if not models_loaded:
        raise HTTPException(
            status_code=500,
            detail="Models are not loaded"
        )

    model_name = data.model.lower().strip()

    if model_name == "random forest":
        selected_model = random_forest_model
        selected_model_name = "Random Forest"

    elif model_name == "decision tree":
        selected_model = decision_tree_model
        selected_model_name = "Decision Tree"

    elif model_name == "xgboost":
        selected_model = xgboost_model
        selected_model_name = "XGBoost"

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model. Choose one of: "
                "Random Forest, Decision Tree, XGBoost"
            )
        )

    features = get_features(data)

    result = format_prediction(
        selected_model,
        features
    )

    return {
        "model": selected_model_name,
        "input": {
            "Temperature": data.Temperature,
            "Humidity": data.Humidity,
            "Wind_Speed": data.Wind_Speed,
            "Cloud_Cover": data.Cloud_Cover,
            "Pressure": data.Pressure
        },
        **result
    }