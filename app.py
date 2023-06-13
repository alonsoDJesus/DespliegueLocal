import uvicorn
from fastapi import FastAPI
from BankNotes import BankNotes
import numpy as numpy
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.get("/")
def index():
    return{"mensaje": "Hola, bienvenido al modelo"}

@app.get("/bienvenida")
def fun_nombre(name: str):
    return{"Hola bienvenido": f"{name}"}

@app.post("/predict")
def predict_banknote(data: BankNotes):
    data = data.dict()
    variance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]
    entropy = data["entropy"]

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])

    if(prediction[0] > 0.5):
        prediction = "Nota Falsa"

    else:
        prediction = "Es nota de banco"
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)