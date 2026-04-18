from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document
from src.config import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL, CHROMA_PERSIST_DIR

# Free local embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Persistent vector DB (free, local)
vectorstore = Chroma(
    collection_name="tickets",
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
)

# Free LLM via Groq
llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.2)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
    return_source_documents=True,
)


def add_ticket_to_vectorstore(ticket_id: int, text: str, metadata: dict):
    doc = Document(page_content=text, metadata={"ticket_id": ticket_id, **metadata})
    vectorstore.add_documents([doc])


def query_rag(question: str) -> dict:
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [d.metadata for d in result["source_documents"]],
    }
