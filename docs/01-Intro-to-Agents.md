---
layout: default
title: "Intro to Agents"
---

# Intro to Agents

## Overview

This lesson introduces AI agents - what they are, why they matter, and where they excel (and struggle). We'll also set up our first basic agent.

## What is an Agent?

**An agent is an LLM that can take actions in a loop until a task is complete.**

Three key parts:

1. **LLM** - A language model that can reason and make decisions
2. **Actions** - The ability to do things (call tools, write files, make API calls)
3. **Loop** - Keeps going until the job is done, not just one response

A chatbot responds once. An agent keeps working.

## Why Agents?

LLMs alone are limited to:
- Knowledge from training data (stale, incomplete)
- Single-turn responses (no persistence)
- Text generation (no real-world impact)

Agents can:
- Access live data (APIs, databases, web)
- Work through multi-step problems
- Actually DO things (create files, send emails, deploy code)
- Recover from errors and try different approaches

## What Agents Are Good At

1. **Repetitive knowledge work** - Research, summarization, data entry
2. **Code generation and modification** - Writing, debugging, refactoring
3. **Multi-step workflows** - Tasks that require several tools in sequence
4. **Exploration tasks** - "Find all the X in this codebase and do Y"

## What Agents Are Bad At

1. **High-stakes decisions without oversight** - Don't let agents approve loans
2. **Creative work requiring human taste** - They can draft, humans should decide
3. **Tasks with ambiguous success criteria** - "Make this better" without specifics
4. **Real-time or latency-sensitive operations** - LLM calls are slow

The biggest failure mode: **agents confidently doing the wrong thing**.

## The Agent Loop

Every agent follows the same basic pattern:

```
1. Receive task
2. Think about what to do
3. Take an action (or respond)
4. Observe the result
5. If not done, go to step 2
```

## Code

### src/agent/run.py

Create the basic agent runner. This is a standalone script you can run directly:

```python
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
```

This is the simplest possible "agent" - really just an LLM call. It's not a true agent yet because:
- No tools (can't take actions)
- No loop (responds once and stops)
- No memory (doesn't use conversation history)

We'll add these capabilities in the following lessons.

## Key Points

1. **Agents = LLM + Actions + Loop** - The model decides what to do and keeps going
2. **The LLM controls the flow** - It's not just responding, it's driving
3. **Start simple** - A basic LLM call is the foundation, we'll add capabilities incrementally

## Exercises

1. **Run the agent** - `python -m src.agent.run` and see the response
2. **Change the model** - Try different models and compare responses
3. **Modify the system prompt** - See how it changes agent behavior
