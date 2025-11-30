import pickle
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model_path = "./model/boosting.pkl"

class AQI(BaseModel):
    lag_1: float
    lag_2: float
    roll_3: float
    roll_6: float

with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.get("/")
def read_root():
    return {"Health status:": "OK"}

@app.post("/predict")
def predict(data:AQI):
    features = np.array([[data.lag_1, data.lag_2, data.roll_3, data.roll_6]])
    pred = model.predict(features)[0]

    # Optional safety clamp: no negative AQI
    pred = max(pred, 0)

    return {"predicted_AQI": round(float(pred), 2)}