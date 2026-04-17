# Building AI Agents — Python

A Python CLI AI agent built from scratch. Python port of [agents-v2](https://github.com/sivakarasala/agents-v2) (TypeScript), built as the companion code repo for the [Building AI Agents](https://sivakarasala.github.io/building-ai-agents/) book.

<a href="https://sivakarasala.github.io/agents-v2-python/" target="_blank">📖 Course Notes (GitHub Pages)</a>

## What it is

A real, working AI agent — not a wrapper around someone else's framework. It has:

- A streaming agent loop (chat completions with SSE)
- Function-calling tools (read/write/list/delete file, shell, code execution, web search)
- Single-turn and multi-turn evaluations with an LLM-as-judge
- Context window management with summarization-based compaction
- Human-in-the-loop tool approval gating
- A Rich + Prompt Toolkit terminal UI

## Setup

```bash
git clone https://github.com/sivakarasala/building-ai-agents-python.git
cd building-ai-agents-python

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# then edit .env and put your OPENAI_API_KEY in
```

## Run the agent

`pip install -e .` (above) registers an `agi` command on your `$PATH` (inside the venv) via the `[project.scripts]` entry in `pyproject.toml`. So you can run it either way:

```bash
# Option 1 — module form (always works, no install needed)
python -m src.main

# Option 2 — installed CLI entrypoint (after `pip install -e .`)
agi
```

Want it available globally like `npm i -g`? Use [pipx](https://pipx.pypa.io/), the Python equivalent for installing CLI apps in isolated environments:

```bash
pipx install .
agi   # now on $PATH everywhere, no venv activation needed
```

## Run the test suite

```bash
pytest
```

The pytest suite is offline — it covers tools, context management, message filtering, the registry, and eval helpers without making any API calls.

## Run an eval (requires `OPENAI_API_KEY`)

```bash
python -m evals.file_tools_eval
python -m evals.shell_tools_eval
python -m evals.agent_multiturn_eval
```

## Layout

```
src/
├── main.py                       # entry point
├── types.py                      # ToolCallInfo, AgentCallbacks, etc.
├── agent/
│   ├── run.py                    # the agent loop
│   ├── execute_tool.py
│   ├── system/                   # system prompt + message filtering
│   ├── context/                  # token estimation + compaction
│   └── tools/                    # file, shell, code execution, web search
├── ui/                           # Rich + Prompt Toolkit CLI
└── ...
evals/
├── executors.py                  # single-turn + multi-turn runners
├── evaluators.py                 # tools_selected / tool_order / llm_judge
├── data/                         # eval datasets
└── mocks/                        # mock tool factories
tests/                            # offline pytest suite
```

## Branches (course progression)

The `done` branch is the complete app. Each `XX-lesson-name` branch contains the solution for the **previous** lesson — students start fresh on a lesson branch and build forward to match the next branch.

| Branch | Lesson |
|---|---|
| `done` | Complete app |
| `09-hitl` | Human-in-the-loop approval |
| `08-shell-tool` | Shell + code execution |
| `07-web-search-context-management` | Web search + context window management |
| `06-file-system-tools` | Full file tools (write/edit/delete) |
| `05-multi-turn-evals` | Multi-turn evals + LLM judge |
| `04-the-agent-loop` | The streaming agent loop |
| `03-single-turn-evals` | Single-turn eval framework |
| `02-tool-calling` | Tool definitions + tool calling |
| `01-intro-to-agents` | First LLM call |

## License

MIT
