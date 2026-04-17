from openai import OpenAI
from dotenv import load_dotenv
from src.agent.system.prompt import SYSTEM_PROMPT

load_dotenv()

MODEL_NAME = "gpt-5-mini"


def run_agent(user_message: str):
    client = OpenAI()
    response = client.responses.create(
        model=MODEL_NAME,
        input=user_message,
        instructions=SYSTEM_PROMPT,
    )

    print(response.output_text)


run_agent("Who won the 2025 IPL final")
