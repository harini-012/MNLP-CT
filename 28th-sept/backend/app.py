from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face model
sentiment_model = pipeline(
    "sentiment-analysis"
)


class Feedback(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "College Feedback NLP API is running"
    }


@app.post("/analyze")
def analyze_feedback(feedback: Feedback):

    result = sentiment_model(feedback.text)[0]

    return {
        "text": feedback.text,
        "sentiment": result["label"],
        "confidence": round(result["score"] * 100, 2)
    }
