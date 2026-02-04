from fastapi import FastAPI
import pandas as pd

app = FastAPI(title="Iris Sepal Predictor")

def dummy_model(width):
    return width * 2 + 1  

@app.get("/predict")
def predict(sepal_width: float):
    prediction = dummy_model(sepal_width)
    return {"sepal_length_pred": prediction}
