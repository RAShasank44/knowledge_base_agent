import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import os

FILE_PATH = "knowledge_base.txt"

# -----------------------------
# Load Knowledge Base (CACHE OK)
# -----------------------------
@st.cache_resource
def load_kb():
    if not os.path.exists(FILE_PATH):
        st.error(f"❌ File not found: {FILE_PATH}")
        st.stop()

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    qa_pairs = re.findall(r"Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)", text, re.S)

    faq = [{"question": q.strip(), "answer": a.strip()} for q, a in qa_pairs]

    return faq

# -----------------------------
# Load Embedding Model (CACHE OK)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Build FAISS Index (NO CACHE)
# -----------------------------
def build_index(faq, model):
    questions = [item["question"] for item in faq]
    embeddings = model.encode(questions).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, questions

# -----------------------------
# Get Answer
# -----------------------------
def get_answer(query, faq, model, index):
    q_embed = model.encode([query]).astype("float32")
    distances, indices = index.search(q_embed, 1)
    return faq[indices[0][0]]["answer"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 Knowledge Base FAQ Assistant")

faq_data = load_kb()
model = load_model()
index, questions = build_index(faq_data, model)

query = st.text_input("Ask your question:")

if query:
    answer = get_answer(query, faq_data, model, index)
    st.subheader("Answer:")
    st.write(answer)
