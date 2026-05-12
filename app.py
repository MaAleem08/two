
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from utils import convert_lower
import __main__
from fastapi.middleware.cors import CORSMiddleware


__main__.convert_lower = convert_lower


# load model
model = joblib.load('model.pkl')

# init app
app = FastAPI(
    title='Diabetes Prediction',
    description='Predict diabetes using GNB',
    version='1.0'
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class person(BaseModel):
    Pregnancies : float
    Glucose	: float
    BloodPressure : float
    SkinThickness : float
    Insulin : float
    BMI	: float
    DiabetesPedigreeFunction : float
    Age : float

# @app.get("/")
# def u():
 #   return "hello "
# prediction
@app.post('/predict')
def predict(data:person):
    df = pd.DataFrame([data.dict()])

    pred = int(model.predict(df)[0])
    prob = model.predict_proba(df)[0]
    
    label = "Yes Diabetic" if pred == 1 else "Not Diabetic"
    
    return {
            'prediction' : pred,
            'Diabetic' : label,
            'probability of Diabetic' : round(float(prob[1]),3)*100 ,
            'probability of Not Diabetic' : round(float(prob[0]),3)*100
            }






