# Project Kickstart — A Multi-Agent System

Automates the first draft of post-award contract analysis: reads a Statement of Work (SOW) and produces a **Start Pack** — extracted obligations, a draft project plan, and flagged risks — for a human project manager to review and approve.

**Platform:** Local (Python + Ollama) · **Collaboration:** GitHub

## Why

Project managers currently read every SOW by hand to identify deliverables, deadlines, and payment terms, then build a plan around them. This is slow, repetitive, and error-prone — problems in the contract (vague terms, missing acceptance criteria, contradicting dates) often surface late, after work has started.

## How it works

Three agents — **Extractor**, **Planner**, **Checker** — coordinated through shared SQLite memory and a single orchestrator. No agent calls another directly; all communication happens by writing to and reading from memory.

- **Extractor** — reads the SOW, pulls out obligations (deliverables, milestones, payments, deadlines), tags each with a source section and confidence level.
- **Planner** — turns approved obligations into a phased project plan. Validated with plain code (not the model): every deliverable covered, every payment scheduled before it, no circular dependencies.
- **Checker** — flags risks and gaps (missing acceptance criteria, contradicting dates, ambiguous wording). Writes a recheck request back to memory rather than fixing issues itself.
- **Orchestrator** — sequences all three agents, logs every run, routes low-confidence records to a human review queue, and renders the final Start Pack.

## Stack

- Python 3.11+
- [Ollama](https://ollama.com) for local model inference (`llama3.1:8b`, also testing `qwen2.5:7b`)
- `pydantic` for structured output validation (with retry on failure)
- SQLite for shared memory, accessed only through `memory.py`

## Repo structure

```
memory.py           # sole database access point
schema.sql          # memory schema
agents/
  extractor.py
  planner.py
  checker.py
orchestrator.py      # runs the pipeline end-to-end
ui/                   # human review interface
tests/
  sow_samples/        # 4 dummy SOWs + answer keys + planted problems
eval.py               # scores results against the answer keys
README.md
NOTES.md              # reflection: what worked, what didn't
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

## Running it

```bash
python orchestrator.py tests/sow_samples/sow1.txt
```

## Team

| Role | Owner |
|---|---|
| Extractor | Naman |
| Planner | Shanaya |
| Checker | Kabir |
| Orchestrator & memory | Priyanka |
| Eval & test contracts | Anirudh |

## Status

🚧 In progress — built for the GigaAcademy Project Kickstart assignment (Aug 17–20, 2026).