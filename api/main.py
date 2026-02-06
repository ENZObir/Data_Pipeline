from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import mlflow.pyfunc
import psycopg2
import os
from pathlib import Path
from dotenv import load_dotenv

# ===============================
# LOAD ENV
# ===============================
ROOT_ENV = Path(__file__).parent.parent / ".env"
if ROOT_ENV.exists():
    load_dotenv(ROOT_ENV)

# ===============================
# CONFIG
# ===============================
MLFLOW_MODEL_PATH = os.getenv("MLFLOW_MODEL_PATH")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# PostgreSQL config
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

# ===============================
# INIT API
# ===============================
app = FastAPI(
    title="Iris Predictor API",
    version="2.0"
)

model = None

# Allow UI (and other local clients) to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ===============================
# LOAD MODEL AT STARTUP
# ===============================
@app.on_event("startup")
def load_model():
    global model

    if not MLFLOW_MODEL_PATH:
        print("⚠️ Aucun modèle MLflow défini dans .env")
        return

    try:
        model = mlflow.pyfunc.load_model(MLFLOW_MODEL_PATH)
        print(f"✅ Modèle MLflow chargé depuis : {MLFLOW_MODEL_PATH}")
    except Exception as e:
        print("⚠️ Impossible de charger le modèle:", e)
        model = None

# ===============================
# FALLBACK DUMMY MODEL
# ===============================
def dummy_model(width: float):
    return width * 2 + 1

# ===============================
# ROUTES
# ===============================
@app.get("/")
def root():
    return {"message": "API running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.get("/test-db")
def test_db():
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        conn.close()
        return {"db_connection": "OK"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/model-status")
def model_status():
    if model:
        return {"status": "loaded", "model_path": MLFLOW_MODEL_PATH}
    else:
        return {"status": "not loaded"}

@app.get("/predict")
def predict(sepal_width: float):

    if sepal_width <= 0:
        raise HTTPException(status_code=400, detail="Invalid input")

    if model:
        try:
            input_df = pd.DataFrame({"sepal_width": [sepal_width]})
            prediction = model.predict(input_df)[0]
            model_used = "mlflow"
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        prediction = dummy_model(sepal_width)
        model_used = "dummy"

    return {
        "input": sepal_width,
        "prediction": float(prediction),
        "model_used": model_used
    }
