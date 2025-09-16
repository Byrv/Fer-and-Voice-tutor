import streamlit as st
import requests
import speech_recognition as sr
import tempfile
import os

st.set_page_config(page_title="Multi-Model Chatbot", layout="centered")
st.title("🧠 Chatbot with Claude & LLaMA3")

# Model selector
model_choice = st.selectbox("Choose model", ["claude", "llama"])
API_URL = f"http://localhost:8000/{model_choice}/invoke"

def send_to_backend(message):
    response = requests.post(API_URL, json={"input": message})
    return response.json().get("output", "No response from backend.")

# Text input
user_input = st.text_input("Type your message:")
if user_input:
    with st.spinner("Thinking..."):
        response = send_to_backend(user_input)
        st.markdown(f"**{model_choice.capitalize()} Bot:** {response}")
