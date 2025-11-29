import streamlit as st

# Load knowledge base
def load_kb(file_path="knowledge_base.txt"):
    kb = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            if ":" in line:
                key, value = line.split(":", 1)
                kb[key.strip().lower()] = value.strip()
    return kb

kb = load_kb()

# Simple keyword-based answer function
def answer_question(question):
    q = question.lower()

    # Direct keyword match
    for key in kb:
        if key in q:
            return kb[key]

    # Word-level partial match
    for key in kb:
        for w in q.split():
            if w in key:
                return kb[key]

    return "Sorry, I don't have information about that."

# Streamlit UI
st.title("📘 Knowledge-Based FAQ Assistant")
st.write("Ask any question based on the knowledge base.")

question = st.text_input("Enter your question:")

if question:
    answer = answer_question(question)
    st.success(answer)
