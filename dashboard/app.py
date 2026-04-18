import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import func
from src.database import get_session, Ticket
from src.rag_engine import query_rag

st.set_page_config(page_title="Support Analytics", page_icon="🎧", layout="wide")
st.title("🎧 Customer Support Analytics Platform")
st.caption("Powered by Groq + ChromaDB + HuggingFace (100% free stack)")

session = get_session()
tickets = session.query(Ticket).filter(Ticket.processed == 1).all()
session.close()

if not tickets:
    st.warning("No processed tickets yet. Run `python -m src.scheduler` first.")
    st.stop()

df = pd.DataFrame(
    [
        {
            "id": t.id,
            "customer": t.customer_id,
            "intent": t.intent,
            "sentiment": t.sentiment,
            "score": t.sentiment_score,
            "summary": t.summary,
            "date": t.created_at,
        }
        for t in tickets
    ]
)

# ─── METRICS ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Tickets", len(df))
c2.metric("Negative %", f"{(df['sentiment']=='negative').mean()*100:.1f}%")
c3.metric("Unique Customers", df["customer"].nunique())
c4.metric("Top Intent", df["intent"].mode()[0] if len(df) else "-")

# ─── CHARTS ───────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Sentiment Distribution")
    fig = px.pie(df, names="sentiment", hole=0.4,
                 color_discrete_map={"negative": "#ff4b4b", "positive": "#00cc96", "neutral": "#636efa"})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🏷️ Top Intents")
    intent_counts = df["intent"].value_counts().reset_index()
    intent_counts.columns = ["intent", "count"]
    fig = px.bar(intent_counts, x="intent", y="count", color="intent")
    st.plotly_chart(fig, use_container_width=True)

# ─── RAG CHAT ─────────────────────────────────────────────
st.subheader("💬 Ask Questions About Your Tickets")
question = st.text_input("e.g. What are the top complaints this week?")
if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        result = query_rag(question)
        st.success(result["answer"])
        with st.expander("📚 Sources"):
            for src in result["sources"][:5]:
                st.json(src)

# ─── TICKET TABLE ─────────────────────────────────────────
st.subheader("📋 Recent Tickets")
st.dataframe(df[["id", "customer", "intent", "sentiment", "summary"]], use_container_width=True)
