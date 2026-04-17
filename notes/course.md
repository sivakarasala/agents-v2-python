# Build an AI Agent from Scratch (Python)

Two days building a general-purpose agent from first principles—tool calls, loops, memory, evals, and guardrails.

## About this Course

Build a general-purpose agent from scratch in Python. No frameworks, no magic—just you, an LLM API, a tool-aware loop, and the patterns that make agents reliable across files, web search, code execution, and beyond.

You'll build a generic agent that can orchestrate tools on your machine: reading and transforming files, calling 3rd-party tools like web search and code execution, and handing results off to the user. The focus is the agent core: a loop that maintains conversation history, uses tool calling to select tools and arguments, updates messages based on tool results, and decides when to stop. Along the way, you'll learn how to manage context with summarization and retrieval, layer in evals to catch failures, and add guardrails and human-in-the-loop checks for sensitive actions.

### What You'll Learn

- Understand the core primitives of an agent: models, tools, history, memory, and orchestration
- Implement a tool-calling loop with the OpenAI Responses API
- Plug in filesystem tools, 3rd-party tools (web search, code execution), and other SDKs with minimal extra code
- Manage context windows with summarization and retrieval so your agent can use local files and search results without blowing the token limit
- Design and run evals that cover both single-step decisions and full multi-step runs
- Add guardrails and human approvals around risky tools like shell commands or bulk edits

### Prerequisites

- Comfortable with Python
- Can run a local dev environment and manage environment variables
- An OpenAI API key
- Basic familiarity with the command line is helpful but not required

## Table of Contents

- [[01-Intro-to-Agents]]
- [[02-Tool-Calling]]
- [[03-Single-Turn-Evals]]
- [[04-The-Agent-Loop]]
- [[05-Multi-turn-Evals]]
- [[06-File-System-Tools]]
- [[07-Web-Search-Context-Management]]
- [[08-Shell-Tool]]
- [[09-Human-in-the-Loop]]
