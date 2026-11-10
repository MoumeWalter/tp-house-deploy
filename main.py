import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("regression.joblib")
COLS = list(model.feature_names_in_)

class House(BaseModel):
    size: float
    nb_rooms: int
    garden: int

@app.get("/predict")
def predict_get():
    return {"y_pred": 2}

@app.post("/predict")
def predict(h: House):
    X = pd.DataFrame([[h.size, h.nb_rooms, h.garden]], columns=COLS)
    return {"y_pred": float(model.predict(X)[0])}
