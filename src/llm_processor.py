from transformers import pipeline
from groq import Groq
from src.config import GROQ_API_KEY, LLM_MODEL, SENTIMENT_MODEL

# Local sentiment model (free, runs on CPU)
sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

# Groq client (free API)
groq_client = Groq(api_key=GROQ_API_KEY)


def analyze_sentiment(text: str):
    try:
        result = sentiment_pipe(text[:512])[0]
        return result["label"], float(result["score"])
    except Exception as e:
        return "neutral", 0.0


def classify_intent(text: str) -> str:
    """Use free Groq LLM for intent classification."""
    prompt = f"""Classify this customer support ticket into ONE category:
[billing, technical_issue, account, refund, feature_request, complaint, praise, other]

Ticket: "{text[:500]}"

Return only the category name, nothing else."""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        return "other"


def summarize(text: str) -> str:
    """Free summarization via Groq."""
    if len(text) < 100:
        return text
    prompt = f"Summarize this support ticket in 1 sentence:\n\n{text[:1500]}"
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text[:200]


def process_ticket(text: str) -> dict:
    sentiment, score = analyze_sentiment(text)
    intent = classify_intent(text)
    summary = summarize(text)
    return {
        "sentiment": sentiment,
        "sentiment_score": score,
        "intent": intent,
        "summary": summary,
    }
