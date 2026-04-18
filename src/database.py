from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from src.config import DATABASE_URL

engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String, index=True)
    raw_text = Column(Text)
    clean_text = Column(Text)
    intent = Column(String, nullable=True)
    sentiment = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Integer, default=0)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()
