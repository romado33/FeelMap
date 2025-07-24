# feelmap_model.py
from transformers import pipeline

# Load Hugging Face emotion detection pipeline
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True)

def detect_emotions(text):
    results = emotion_pipeline(text)
    return sorted(results[0], key=lambda x: x['score'], reverse=True)
