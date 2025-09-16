import requests

from aws_session import get_aws_client

# Initialize Polly client
polly_client = get_aws_client()

def synthesize_speech(text, filename="bot_response.mp3"):
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId="Joanna"
    )
    with open(filename, "wb") as file:
        file.write(response["AudioStream"].read())
    return filename


def send_to_backend(message):
    API_URL = f"http://localhost:8000/claude/invoke"
    response = requests.post(API_URL, json={"input": message})
    return response.json().get("output", "No response from backend.")
