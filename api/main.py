from fastapi import FastAPI, HTTPException
import pandas as pd
import mlflow.pyfunc
import psycopg2
import os

# ===============================
# CONFIG
# ===============================

MLFLOW_MODEL_PATH = os.getenv("MLFLOW_MODEL_PATH", None)

POSTGRES_HOST = "postgres"   # IMPORTANT: nom du service docker
POSTGRES_PORT = 5432
POSTGRES_DB = "postgres"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"


# ===============================
# INIT API
# ===============================

app = FastAPI(
    title="Iris Predictor API",
    version="2.0"
)

model = None


# ===============================
# LOAD MODEL AU DEMARRAGE
# ===============================

@app.on_event("startup")
def load_model():
    global model

    if not MLFLOW_MODEL_PATH:
        print("⚠️ Aucun modèle MLflow défini")
        return

    try:
        model = mlflow.pyfunc.load_model(MLFLOW_MODEL_PATH)
        print("✅ Modèle MLflow chargé")
    except Exception as e:
        print("⚠️ Impossible de charger le modèle:", e)
        model = None


# ===============================
# FALLBACK MODEL
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


# 🔥 ROUTE TEST POSTGRES (TEMPORAIRE)
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
