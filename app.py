import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data and embeddings
df = pd.read_csv("symptoms_and_treatments.csv")
symptom_embeddings = np.load("symptom_embeddings.npy")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to find top treatments
def find_top_treatments(query_symptom, top_k=3):
    query_embedding = model.encode([query_symptom])
    similarities = cosine_similarity(query_embedding, symptom_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        similar_symptom = df.iloc[idx]["symptom"]
        treatment = df.iloc[idx]["treatment"]
        similarity_score = similarities[idx]
        results.append((similar_symptom, treatment, similarity_score))
    return results

# Streamlit interface
st.title("Medical Symptom Checker Chatbot")
st.write("Enter a symptom to get treatments.")

query = st.text_input("Enter symptom:")
if query:
    results = find_top_treatments(query)
    st.write("### Results:")
    for similar_symptom, treatment, score in results:
        st.write(f"**Symptom:** {similar_symptom}")
        st.write(f"**Treatment:** {treatment}")
        st.write(f"**Confidence:** {score:.2f}")
        st.write("---")