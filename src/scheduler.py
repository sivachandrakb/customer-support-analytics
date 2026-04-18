import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from src.database import get_session, Ticket, init_db
from src.preprocessing import preprocess
from src.llm_processor import process_ticket
from src.rag_engine import add_ticket_to_vectorstore


def ingest_csv(path: str = "data/sample_tickets.csv"):
    """Load tickets from CSV into DB."""
    df = pd.read_csv(path)
    session = get_session()
    for _, row in df.iterrows():
        ticket = Ticket(
            customer_id=str(row.get("customer_id", "unknown")),
            raw_text=str(row.get("text", "")),
            clean_text=preprocess(str(row.get("text", ""))),
        )
        session.add(ticket)
    session.commit()
    session.close()
    print(f"✅ Ingested {len(df)} tickets")


def process_pending_tickets(batch_size: int = 10):
    """Process unprocessed tickets with LLM."""
    session = get_session()
    pending = session.query(Ticket).filter(Ticket.processed == 0).limit(batch_size).all()
    print(f"🔄 Processing {len(pending)} tickets...")

    for ticket in pending:
        try:
            enriched = process_ticket(ticket.clean_text)
            ticket.sentiment = enriched["sentiment"]
            ticket.sentiment_score = enriched["sentiment_score"]
            ticket.intent = enriched["intent"]
            ticket.summary = enriched["summary"]
            ticket.processed = 1

            add_ticket_to_vectorstore(
                ticket.id,
                ticket.clean_text,
                {
                    "sentiment": enriched["sentiment"],
                    "intent": enriched["intent"],
                    "customer_id": ticket.customer_id,
                },
            )
            session.commit()
            print(f"  ✓ Ticket {ticket.id} → {enriched['intent']} / {enriched['sentiment']}")
        except Exception as e:
            print(f"  ✗ Error on ticket {ticket.id}: {e}")
            session.rollback()
    session.close()


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(process_pending_tickets, "interval", minutes=5)
    scheduler.start()
    print("📅 Scheduler started (processes every 5 min)")


if __name__ == "__main__":
    init_db()
    ingest_csv()
    process_pending_tickets(batch_size=50)
