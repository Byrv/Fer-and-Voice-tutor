import os
from dotenv import load_dotenv
import boto3
from typing import List, Optional
import threading
from langchain_aws.chat_models import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from polly import PollySpeaker 
from rag import RAGMemory
from slowtutor import SlowTutor
from appstate import AppState
from facial_fer import StudyModeFER
# =========================
# Environment & Bedrock
# =========================
load_dotenv()
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")

bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)
voice = PollySpeaker(region="us-east-1", 
                     voice="Kajal", 
                     engine="neural", 
                     lang="en-IN")

# =========================
# Language variants
# =========================
ENGLISH_VARIANTS = {
    "india": "indian_english",
    "in": "indian_english",
    "us": "american_english",
    "usa": "american_english",
    "united states": "american_english",
    "uk": "british_english",
    "england": "british_english"
}

def get_language_instruction(style: str) -> str:
    if style == "indian_english":
        return ("Use Indian English. Be warm, clear, and encouraging. "
                "Use familiar, everyday examples.")
    elif style == "american_english":
        return ("Use American English. Be upbeat, clear, and friendly. "
                "Use simple, everyday examples.")
    elif style == "british_english":
        return ("Use British English. Be polite, clear, and supportive. "
                "Use familiar, everyday examples.")
    else:
        return "Use simple, friendly English."

def get_age_constraints(age: int) -> str:
    if age <= 7:
        return (
            "For a young child:\n"
            "- Use very short sentences (≤ 10–12 words).\n"
            "- One idea per step. No jargon.\n"
            "- Use concrete examples.\n"
            "- Max 3–4 steps."
        )
    elif age <= 10:
        return (
            "For a child:\n"
            "- Use short sentences (≤ 12–14 words).\n"
            "- Explain with simple examples.\n"
            "- Keep 4–5 clear steps."
        )
    elif age <= 13:
        return (
            "For a pre-teen:\n"
            "- Be casual and supportive.\n"
            "- Keep steps concise (up to 5–6 if needed).\n"
            "- Invite thinking aloud."
        )
    else:
        return (
            "For a teenager:\n"
            "- Be a helpful tutor.\n"
            "- Encourage reasoning, show concise steps, avoid jargon.\n"
            "- Offer a quick check and a practice problem."
        )

def clamp_age(value, default=10):
    try:
        a = int(value)
    except:
        return default
    return max(5, min(17, a))


def normalize_choice(text: str) -> str:
    """
    Map user input to normalized choices:
    - 'another hint' → 'hint'
    - 'first step' / 'step' → 'step'
    - 'answer' / 'solution' → 'answer'
    - 'stop' / 'quit' / 'exit' → 'stop'
    - anything else → 'free'
    """
    if not text:
        return "free"
    t = text.strip().lower()
    if any(k in t for k in ["stop", "quit", "exit", "cancel"]):
        return "stop"
    if any(k in t for k in ["answer", "solution", "show answer", "final"]):
        return "answer"
    if any(k in t for k in ["step", "first step", "show step"]):
        return "step"
    if any(k in t for k in ["hint", "another hint", "more hint", "next hint"]):
        return "hint"
    return "free"

# =========================
# Slow Tutor class
# =========================


# =========================
# Main CLI
# =========================


def main():
    # Initialize shared state
    state = AppState()

    # Ask if emotion detection should be enabled
    emotion_enabled = input("🎥 Enable emotion detection via webcam? (y/n, default y): ").strip().lower()
    if emotion_enabled != "n":
        fer = StudyModeFER(state=state)
        fer_thread = threading.Thread(target=fer.run, daemon=True)
        fer_thread.start()

    print("👋 Hi! I’m your AI tutor (Slow Thinking Mode ready).")
    location = input("🌍 Where do you live (e.g., India, USA, UK)? ").strip().lower()
    age_input = input("🎂 How old are you? ").strip()
    slow_mode_input = input("🐢 Turn ON slow thinking mode? (y/n, default y): ").strip().lower()

    age = clamp_age(age_input, default=10)
    language_style = ENGLISH_VARIANTS.get(location, "neutral_english")
    language_instruction = get_language_instruction(language_style)
    age_instruction = get_age_constraints(age)
    slow_mode = False if slow_mode_input == "n" else True
    # LLM setup
    claude_llm = ChatBedrock(
        client=bedrock_runtime,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name=aws_region,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs={
            "temperature": 0.2,
            "max_tokens": 900,
        },
    )

    # Memory
    rag_memory = RAGMemory()

    # Tutor setup
    tutor = SlowTutor(
        llm=claude_llm,
        rag_memory=rag_memory,
        language_instruction=language_instruction,
        age_instruction=age_instruction,
        state=state,
        slow_mode=slow_mode,
        max_hints=3
    )


    while True:
        voice.speak_text("Hello! Polly voice check. If you hear this, T T S is working.")
        question = input("\n❓ What question do you want help with? (or 'exit') ").strip()
        if question.lower() == "exit":
            print("Goodbye! Keep learning.")
            break
        response = tutor.tutor_once(question)
        print(response)
        voice.speak_response(response)   # non-blocking; audio plays while loop continue


if __name__ == "__main__":
    main()
