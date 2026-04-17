import json
from openai import OpenAI
from dotenv import load_dotenv
from src.agent.system.prompt import SYSTEM_PROMPT
from src.agent.tools import ALL_TOOLS
from src.agent.execute_tool import execute_tool

load_dotenv()

MODEL_NAME = "gpt-5-mini"


def run_agent(user_message: str):
    client = OpenAI()
    response = client.responses.create(
        model=MODEL_NAME,
        input=user_message,
        instructions=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
    )

    print(response.output_text)

    for item in response.output:
        item_dict = item.model_dump(exclude_none=True)
        if item_dict.get("type") == "function_call":
            args = json.loads(item_dict.get("arguments") or "{}")
            result = execute_tool(item_dict["name"], args)
            print(f"Tool: {item_dict['name']}, Result: {result}")


run_agent("What is the current date and time?")
