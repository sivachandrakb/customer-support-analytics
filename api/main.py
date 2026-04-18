from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from src.database import get_session, Ticket, init_db
from src.rag_engine import query_rag

app = FastAPI(title="Support Analytics API")
init_db()


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "ok", "service": "Support Analytics API"}


@app.get("/tickets")
def list_tickets(limit: int = 20):
    session = get_session()
    tickets = session.query(Ticket).order_by(Ticket.created_at.desc()).limit(limit).all()
    session.close()
    return [
        {
            "id": t.id,
            "customer_id": t.customer_id,
            "intent": t.intent,
            "sentiment": t.sentiment,
            "summary": t.summary,
            "created_at": t.created_at,
        }
        for t in tickets
    ]


@app.get("/insights/trends")
def sentiment_trends():
    session = get_session()
    data = (
        session.query(Ticket.sentiment, func.count(Ticket.id))
        .filter(Ticket.processed == 1)
        .group_by(Ticket.sentiment)
        .all()
    )
    session.close()
    return {s: c for s, c in data}


@app.get("/insights/intents")
def top_intents():
    session = get_session()
    data = (
        session.query(Ticket.intent, func.count(Ticket.id))
        .filter(Ticket.processed == 1)
        .group_by(Ticket.intent)
        .order_by(func.count(Ticket.id).desc())
        .all()
    )
    session.close()
    return {i: c for i, c in data}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = query_rag(req.question)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
