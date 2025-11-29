import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re


@st.cache_resource
def load_kb():
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Parse Q/A pairs
    qa_pairs = re.findall(r"Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)", text, re.S)

    faq = []
    for q, a in qa_pairs:
        faq.append({"question": q.strip(), "answer": a.strip()})
    
    return faq


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def build_index(faq, model):
    questions = [item["question"] for item in faq]
    embeddings = model.encode(questions).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, questions


def get_answer(query, faq, model, index, questions):
    q_embed = model.encode([query]).astype("float32")
    distances, indices = index.search(q_embed, 1)
    return faq[indices[0][0]]["answer"]


st.title("📚 Knowledge Base FAQ Assistant")
st.write("Ask any question from the knowledge base!")

faq_data = load_kb()
model = load_model()
index, questions = build_index(faq_data, model)

user_query = st.text_input("Enter your question here:")

if user_query:
    answer = get_answer(user_query, faq_data, model, index, questions)
    st.subheader("Answer:")
    st.write(answer)
