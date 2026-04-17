---
layout: default
title: Home
---

# Build an AI Agent from Scratch (Python)

Two days building a general-purpose agent from first principles — tool calls, loops, memory, evals, and guardrails.

## About this Course

Build a general-purpose agent from scratch in Python. No frameworks, no magic — just you, an LLM API, a tool-aware loop, and the patterns that make agents reliable across files, web search, code execution, and beyond.

You'll build a generic agent that can orchestrate tools on your machine: reading and transforming files, calling 3rd-party tools like web search and code execution, and handing results off to the user. The focus is the agent core: a loop that maintains conversation history, uses tool calling to select tools and arguments, updates messages based on tool results, and decides when to stop.

### What You'll Learn

- Understand the core primitives of an agent: models, tools, history, memory, and orchestration
- Implement a tool-calling loop with the OpenAI Responses API
- Plug in filesystem tools, 3rd-party tools (web search, code execution), and other SDKs
- Manage context windows with summarization and retrieval
- Design and run evals that cover both single-step decisions and full multi-step runs
- Add guardrails and human approvals around risky tools

### Prerequisites

- Comfortable with Python
- Can run a local dev environment and manage environment variables
- An OpenAI API key

---

## Lessons

| # | Lesson | Branch |
|---|--------|--------|
| 1 | [Intro to Agents](01-Intro-to-Agents) | `lesson-01` |
| 2 | [Tool Calling](02-Tool-Calling) | `lesson-02` |
| 3 | [Single Turn Evals](03-Single-Turn-Evals) | `lesson-03` |
| 4 | [The Agent Loop](04-The-Agent-Loop) | `lesson-04` |
| 5 | [Multi-turn Evals](05-Multi-turn-Evals) | `lesson-05` |
| 6 | [File System Tools](06-File-System-Tools) | `lesson-06` |
| 7 | [Web Search + Context Management](07-Web-Search-Context-Management) | `lesson-07` |
| 8 | [Shell Tool](08-Shell-Tool) | `lesson-08` |
| 9 | [Human-in-the-Loop](09-Human-in-the-Loop) | `lesson-09` |

---

[View on GitHub](https://github.com/sivakarasala/agents-v2-python)
