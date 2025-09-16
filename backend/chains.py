from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableLambda
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM
import boto3
from langchain_aws import ChatBedrock
from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
env_path = os.path.join("..", ".env")
load_dotenv(dotenv_path=env_path)
print(env_path)

# Access the variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")
print(aws_region)

session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

bedrock_client = session.client("bedrock-runtime")

def log_input(inputs):
    logging.info(f"Claude Chain received input: {inputs}")
    return inputs

# Claude 3 Chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

claude_llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name=aws_region,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
claude_prompt = PromptTemplate.from_template("Human: {input}\nAssistant:")
claude_chain = RunnableLambda(log_input) | claude_prompt | claude_llm

# LLaMA3 Chain
llama_llm = ChatBedrock(
    model_id="meta.llama3-70b-instruct-v1:0",
    region_name=aws_region,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
llama_prompt = PromptTemplate.from_template("User: {input}\nBot:")
llama_chain = RunnableLambda(log_input) | llama_prompt | llama_llm

