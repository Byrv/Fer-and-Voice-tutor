import streamlit as st
import speech_recognition as sr
import tempfile
import os
import time
from datetime import datetime

from utils import synthesize_speech, send_to_backend


# Page config
st.set_page_config(page_title="Noodle Crew", layout="centered")

# Custom CSS for chat layout
with open("frontend/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="ribbon">
        🧠 Noodle Crew – Homework Assistant
    </div>
""", unsafe_allow_html=True)

st.image("frontend/noodle_crew.png", caption="Meet the Noodle Crew!", width='stretch')
st.markdown("<h4 style='text-align:center; color:#555;'>Smart help from the Noodle Crew!</h4>", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Use mic_input to prefill the input field
prefill_text = st.session_state.get("mic_input", "")

# Display chat history
chat_pairs = []
history = st.session_state.chat_history
i = 0
while i < len(history) - 1:
    if history[i]["role"] == "user" and history[i + 1]["role"] == "bot":
        chat_pairs.append((history[i], history[i + 1]))
        i += 2
    else:
        i += 1  # Skip unmatched entries

for idx, (user_msg, bot_msg) in enumerate(chat_pairs):
    st.markdown(
        f'<div class="chat-message user-message">🧑‍💻 {user_msg["text"]}'
        f'<div class="timestamp">{user_msg["time"]}</div></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="chat-message bot-message">🤖 {bot_msg["text"]["content"]}'
        f'<div class="timestamp">{bot_msg["time"]}</div></div>',
        unsafe_allow_html=True
    )

# Play audio for the last bot response only
if chat_pairs and "last_audio_path" in st.session_state:
    st.markdown("**🔈 Listen to the bot's response:**")
    audio_bytes = open(st.session_state["last_audio_path"], "rb").read()
    st.audio(audio_bytes, format="audio/mp3")


# Spacer to push input to bottom
st.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)

# Input section at the bottom
with st.container():
    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        user_input = st.text_area(
            "Type or speak your message:",
            value=prefill_text,
            key="user_input",
            height=100,
            max_chars=None
        )

    with col2:
        mic_clicked = st.button("🎤", help="Speak your message")

    with col3:
        send_clicked = st.button("➤", help="Send message")

# Microphone input
if mic_clicked:
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 3

    try:
        with sr.Microphone() as source:
            status = st.empty()
            status.info("🎙️ Listening... Please speak clearly.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)
            status.info("✅ Done Listening")
            status.empty()

            transcribed_text = recognizer.recognize_google(audio)
            st.success(f"You said: {transcribed_text}")
            st.session_state["mic_input"] = transcribed_text
            st.session_state["mic_ready"] = True
            st.rerun()

    except sr.WaitTimeoutError:
        st.error("⏱️ Listening timed out. Please try again.")
    except sr.UnknownValueError:
        st.error("🤷 Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"🔌 Speech recognition error: {e}")
    except OSError as e:
        st.error(f"🎤 Microphone error: {e}")

# Process input when Send is clicked
if send_clicked and user_input:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append({"role": "user", "text": user_input, "time": timestamp})
    with st.spinner("Thinking..."):
        response = send_to_backend(user_input)
        #print(f'Bot response: {response}')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({"role": "bot", "text": response, "time": timestamp})

        bot_text = response["content"] if isinstance(response, dict) and "content" in response else str(response)
        audio_path = synthesize_speech(bot_text)
        st.session_state["last_audio_path"] = audio_path

    st.session_state["mic_input"] = ""
    st.rerun()
