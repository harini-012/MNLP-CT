from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import re

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face model
sentiment_model = pipeline("sentiment-analysis")


class Feedback(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "College Feedback NLP API is running"
    }


@app.post("/analyze")
def analyze_feedback(feedback: Feedback):

    text = feedback.text

    # Sentiment
    result = sentiment_model(text)[0]

    sentiment = result["label"]
    confidence = round(result["score"] * 100, 2)

    # Word count
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    # Keywords
    keyword_list = []

    keywords_to_check = [
        "faculty",
        "teacher",
        "teaching",
        "explains",
        "class",
        "subject",
        "course",
        "lab"
    ]

    for word in keywords_to_check:
        if word.lower() in text.lower():
            keyword_list.append(word.capitalize())

    keywords = ", ".join(keyword_list)

    # Category
    category = "General"

    teaching_words = [
        "faculty",
        "teacher",
        "teaching",
        "explains",
        "class",
        "subject",
        "course"
    ]

    if any(word in text.lower() for word in teaching_words):
        category = "Teaching"

    # Emotion
    if sentiment == "POSITIVE":
        emotion = "Satisfaction"
    else:
        emotion = "Dissatisfaction"

    # Recommendation
    if sentiment == "POSITIVE":
        recommendation = "Appreciative feedback"
    else:
        recommendation = "Consider improvement"

    # Priority
    if sentiment == "NEGATIVE":
        priority = "High"
    else:
        priority = "Low"

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "category": category,
        "emotion": emotion,
        "keywords": keywords,
        "word_count": word_count,
        "recommendation": recommendation,
        "priority": priority
    }
