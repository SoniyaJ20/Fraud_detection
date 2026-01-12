from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the files created by train_aegis.py
model = joblib.load('aegis_model.joblib')
scaler = joblib.load('scaler.joblib')

class Transaction(BaseModel):
    v_features: list
    amount: float

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(data: Transaction):
    try:
        # 1. Create a list of column names exactly as they were during training
        # V1, V2, ... V28
        columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # 2. Scale the amount
        scaled_amount = scaler.transform([[data.amount]])[0][0]
        
        # 3. Combine V-features and the scaled amount
        full_row = data.v_features + [scaled_amount]
        
        # 4. CRITICAL FIX: Convert to DataFrame with column names
        X_input = pd.DataFrame([full_row], columns=columns)
        
        # 5. Predict
        prediction = model.predict(X_input)
        
        verdict = "FRAUD DETECTED" if prediction[0] == -1 else "SECURE"
        return {"verdict": verdict}
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"verdict": "Error processing data", "error": str(e)}