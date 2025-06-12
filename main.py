from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

# ✅ Allow all origins, methods, headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextData(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextData):
    vec = vectorizer.transform([data.text])
    pred = model.predict(vec)
    return {"sentiment": "positive" if pred[0] == 1 else "negative"}
