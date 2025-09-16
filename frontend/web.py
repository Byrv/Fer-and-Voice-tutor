import streamlit as st
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import tempfile
import os
import time
from datetime import datetime

from utils import synthesize_speech, send_to_backend


from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

# Dummy emotion classifier (replace with real model later)
def classify_emotion(face_img):
    # Placeholder logic: always return "Neutral"
    return "Neutral"

# Emotion detection transformer
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            emotion = classify_emotion(face_img)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img


# Page config
st.set_page_config(page_title="Noodle Crew", layout="centered")

# Custom CSS for chat layout
with open("frontend/style.css") as f:
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


# Ask for webcam permission and start detection
#st.markdown("### 🎥 Enable Webcam for Emotion Detection")
#webrtc_streamer(key="emotion-detect", video_transformer_factory=EmotionDetector)


# First-time launch questions (must be filled and sent to backend)
if "user_location" not in st.session_state or "user_age" not in st.session_state:
    st.markdown("### 👋 Welcome! Let's get to know you a bit.")

    location = st.text_input("🌍 Where do you live (e.g., India, USA, UK)?")
    age = st.text_input("🎂 How old are you?")

    # Validate non-empty and correct type
    if isinstance(location, str) and location.strip() and isinstance(age, str) and age.strip():
        st.session_state.user_location = location.strip()
        st.session_state.user_age = age.strip()
        try:
            backend_response = send_to_backend(st.session_state.user_location)
            time.sleep(1)
            backend_response = send_to_backend(st.session_state.user_age)
            st.success("✅ Thanks! You're all set.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to send info to server: {e}")
        st.rerun()

    else:
        st.warning("⚠️ Please fill in both fields to continue.")
        st.stop()


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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({"role": "bot", "text": response, "time": timestamp})

        bot_text = response["content"] if isinstance(response, dict) and "content" in response else str(response)
        audio_path = synthesize_speech(bot_text)
        st.session_state["last_audio_path"] = audio_path

    st.session_state["mic_input"] = ""
    st.rerun()
