import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tickets.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Free models
LLM_MODEL = "llama-3.1-8b-instant"  # Free on Groq
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local, free
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Local, free
