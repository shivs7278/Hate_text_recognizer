from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model & tokenizer
model = load_model("lstm_model.h5")
tokenizer = joblib.load("lstm_tokenizer.pkl")
max_len = 50

class PredictRequest(BaseModel):
    text: str
    model: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(req: PredictRequest):
    text = req.text

    # Tokenize and pad input
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")

    # Predict
    pred = model.predict(padded)
    label = np.argmax(pred[0])

    label_map = {
        0: "Hate Speech Detected",
        1: "Offensive Language Detected",
        2: "Clean Text Detected"
    }

    return {
        "model": "LSTM",
        "result": label_map.get(label, "Unknown")
    }
