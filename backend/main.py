from fastapi import FastAPI, Request
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import boto3
import json
from langserve import add_routes
import uvicorn
from backend.chains import claude_chain, llama_chain


app = FastAPI(
    title="FastAPI Server",
    version="1.0",
    description="API server"
)

# Load environment variables from .env file
env_path = os.path.join("..", ".env")
load_dotenv(dotenv_path=env_path)


add_routes(app, claude_chain, path="/claude")
add_routes(app, llama_chain, path="/llama")


"""llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-east-1"
)"""

"""
class ChatRequest(BaseModel):
    message: str

def query_bedrock(model_id, prompt):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']

@app.post("/chat/claude")
async def chat_claude(req: ChatRequest):
    return {"response": query_bedrock("anthropic.claude-3-5-sonnet-20240620-v1:0", req.message)}

@app.post("/chat/llama")
async def chat_llama(req: ChatRequest):
    return {"response": query_bedrock("meta.llama3-70b-instruct-v1:0", req.message)}
"""





















"""
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"  # Or use LLaMA3 model ID

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    prompt = req.message
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response['body'].read())
    reply = response_body['content'][0]['text']
    return {"response": reply}
"""