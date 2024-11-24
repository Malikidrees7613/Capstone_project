import streamlit as st
import numpy as np
import pandas as pd
from sentencetransformers import SentenceTransformer
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
        similar_symptom = df.iloc[idx]["Symptoms"]
        treatment = df.iloc[idx]["Treatments"]
        similarity_score = similarities[idx]
        results.append((similar_symptom, treatment, similarity_score))
    return results

# Streamlit interface
st.title("Medical Symptom Checker Chatbot")
st.write("Hello! I'm your virtual assistant. Let me ask you a few questions to help diagnose your symptoms.")

# Question 1: Ask about the main symptom
symptom = st.text_input("What symptom are you experiencing?")

if symptom:
    st.write(f"Got it! You're feeling {symptom}. Let me ask a few more questions.")
    
    # Question 2: Duration of the symptom
    duration = st.text_input("How long have you been feeling this symptom? (e.g., a few days, weeks, etc.)")
    
    if duration:
        st.write(f"Thanks for the information. You've been feeling {symptom} for {duration}.")
        
        # Question 3: Additional symptoms
        additional_symptoms = st.text_input("Have you experienced any other symptoms, like fever, dizziness, etc.?")
        
        if additional_symptoms:
            st.write(f"Noted! You also have the following symptoms: {additional_symptoms}.")
            
            # Combine all symptoms into a single query for treatment recommendations
            combined_symptoms = f"{symptom}, {additional_symptoms}"
            
            # Get top treatments
            results = find_top_treatments(combined_symptoms)
            
            # Display results in a conversational manner
            st.write("Based on what you've told me, here are some possible treatments:")
            for similar_symptom, treatment, score in results:
                st.write(f"**Symptom:** {similar_symptom}")
                st.write(f"**Treatment:** {treatment}")
                st.write(f"**Confidence:** {score:.2f}")
                st.write("---")
            
            # Final Doctor's response (summarized)
            st.write("It seems like you might be dealing with some common symptoms related to cold or flu, but a proper diagnosis requires consultation with a healthcare professional.")
        else:
            st.write("I suggest monitoring your symptoms closely. If the condition persists, you may need to consult a doctor.")
else:
    st.write("Please start by telling me your main symptom to get started!")
