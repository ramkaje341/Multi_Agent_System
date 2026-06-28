# 🏥 Multi Agent System

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-6f42c1)](#architecture--workflow)
[![Framework](https://img.shields.io/badge/Framework-LangGraph%20%2B%20LangChain-0ea5e9)](#key-features)

A Python multi-agent system for clinical decision-support workflows.  
It combines intent routing, specialist agents, tool-calling, retrieval-augmented generation (RAG), and a Streamlit interface into one end-to-end application.

> ⚠️ This project is intended for decision support and experimentation. Outputs should always be reviewed by qualified professionals.

## Project Overview

This repository implements a coordinated **multi-agent pipeline** where an orchestrator routes a query to specialist agents (diagnosis, drug, literature, image, and web search), then synthesizes responses into a final answer.

The system integrates:
- **LangGraph** for orchestration/state flow
- **LangChain tool-calling** for MCP-style agent tools
- **ChromaDB + sentence-transformers** for local RAG context
- **Streamlit** for interactive usage
- **SQLite (SQLAlchemy)** for session memory and trace logging

## Key Features

- Multi-agent orchestration with intent classification and routing
- Specialist agents for diagnosis, drug analysis, literature, and image analysis
- Tool-calling layer for PubMed, OpenFDA, coding systems (ICD/SNOMED/RxNorm), and Tavily search
- Confidence-aware web fallback when local/tool responses are weak
- Local RAG pipeline with ingestion and persistent vector storage
- Session memory and traceability with SQLite-backed history
- Streamlit UI with chat, examples, upload support, and execution traces

## Architecture & Workflow

High-level flow (from `agents/orchestrator.py`):

1. **Classify intent** from user query
2. **Route through specialist nodes** (diagnosis → drug → literature → image)
3. **Optional web search node** for explicit recency needs or low confidence
4. **Summarize** all agent outputs
5. **Reflect/QA pass** (retry summary if needed)
6. Return final clinician + patient-friendly outputs

Tools are defined in `mcp/tool_definitions.py` and executed through an internal tool-call loop.

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1) Configuration check

```bash
python run.py --check
```

### 2) Ingest sample/local documents

```bash
python ingest.py --sample
# or
python ingest.py --dir data/sample_docs
```

### 3) Start the application

```bash
python run.py
```

This launches the Streamlit UI (default: `http://localhost:8501`).

You can also run Streamlit directly:

```bash
streamlit run ui/app.py
```

## Configuration

Create a `.env` file in the repository root and set relevant keys.

Core keys used by the system include:
- `GROQ_API_KEY` (required for main LLM pipeline)
- `GEMINI_API_KEY` (used for image-capable workflows)
- `TAVILY_API_KEY` (used for web search fallback)

Optional settings are exposed in `config/settings.py` (e.g., vector DB path, top-k retrieval, logging level, memory DB path).

## Project Structure

```text
Multi_Agent_System/
├── agents/          # Orchestrator + specialist agents
├── config/          # Environment/config constants
├── mcp/             # Tool definitions and execution loop
├── memory/          # SQLite-backed memory + traces
├── rag/             # Embeddings, retrieval, vector store
├── tools/           # API/tool integrations (PubMed, OpenFDA, Tavily, vision)
├── ui/              # Streamlit frontend
├── ingest.py        # Document ingestion pipeline
├── run.py           # Main launcher
└── requirements.txt
```

## Examples

Example queries (similar to in-app presets):

- `Patient has pleuritic chest pain, fever, and tachycardia. What are likely differentials?`
- `Patient on warfarin started metronidazole. What interactions should I consider?`
- `What is the latest evidence for SGLT2 inhibitors in heart failure?`
- `Analyze this uploaded chest X-ray and summarize key findings.`

Programmatic usage example:

```python
from agents.orchestrator import run_pipeline

result = run_pipeline(query="Metformin dosing considerations in CKD")
print(result["final_response"])
```

## Roadmap

Potential next steps for the project:

- Add automated test coverage for orchestrator/tool integration paths
- Add containerized deployment workflow
- Add evaluation harness for response quality and safety checks
- Expand observability/telemetry for agent-level performance
- Provide stricter role-based guardrails for production-like environments

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Make focused changes with clear commit messages
4. Open a pull request with context and testing notes

## License

No license file is currently present in this repository.  
If you plan to distribute or reuse this project, add an explicit open-source license (for example, MIT/Apache-2.0) in a top-level `LICENSE` file.
