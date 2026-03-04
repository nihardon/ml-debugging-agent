# ML Experiment Debugging Agent

A full-stack ML debugging tool that accepts training logs, loss curves, configs, and stack traces, then returns a structured, citation-backed diagnostic report using a 3-node LangGraph pipeline.

## Architecture

```
Upload Artifacts
      │
      ▼
┌─────────────────────────────────────────────┐
│              LangGraph Pipeline             │
│                                             │
│  Node 1: Parser      →  SymptomSet          │
│  (pure Python/regex)                        │
│           │                                 │
│  Node 2: Retriever   →  KBDocuments (top 5) │
│  (ChromaDB query)                           │
│           │                                 │
│  Node 3: Advisor     →  DiagnosticReport    │
│  (1× Claude API call)                       │
└─────────────────────────────────────────────┘
      │
      ▼
  JSON Report + Annotated Loss Chart
```

## Setup

### 1. Clone & install Python dependencies

```bash
cd ml_debugging_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Seed the knowledge base

This downloads `all-MiniLM-L6-v2` on first run (≈90MB, cached locally).

```bash
python scripts/seed_kb.py
```

### 4. Start the FastAPI backend

```bash
uvicorn backend.api:app --reload --reload-dir backend --port 8000
```

### 5. Install and start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Usage

1. Upload any combination of:
   - **Loss Curve CSV** — columns: `step`/`epoch` and `loss`
   - **Config** — `.yaml` or `.json` training config
   - **Log file** — `.txt` training log (GPU memory, etc.)
   - **Stack trace** — paste directly into the textarea

2. Click **Run Diagnosis**.

3. The report shows:
   - Status badge (Critical / Warning / Healthy)
   - Root cause + plain-English explanation
   - Runnable PyTorch fix snippet
   - Prioritized action plan
   - Citations from the knowledge base
   - Loss chart with divergence annotation (if applicable)

## API

### `POST /diagnose`
Accepts `multipart/form-data`:
| Field | Type | Description |
|---|---|---|
| `log_file` | File (.txt) | Training log |
| `csv_file` | File (.csv) | Loss curve |
| `config_file` | File (.yaml/.json) | Training config |
| `stack_trace` | String | Pasted stack trace |

Returns a `DiagnosticReport` JSON object.

### `GET /health`
Returns `{ "status": "ok", "kb_doc_count": 35, "kb_seeded": true }`.

## Project Structure

```
backend/
  agents/
    parser.py      # Node 1: pure Python parsing
    retriever.py   # Node 2: ChromaDB retrieval
    advisor.py     # Node 3: Anthropic LLM call
  kb/
    chroma_store.py  # ChromaDB wrapper
  graph.py         # LangGraph DAG
  api.py           # FastAPI app
  models.py        # Pydantic schemas

frontend/
  src/
    App.jsx
    components/
      UploadPanel.jsx
      LossChart.jsx
      DiagnosticReport.jsx

scripts/
  seed_kb.py       # Seed 35 KB entries (idempotent, ID-based)

chroma_db/         # Persisted ChromaDB (created at runtime)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Claude (claude-sonnet-4-20250514) |
| Vector store | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Backend | FastAPI + uvicorn |
| Frontend | React + Vite + Plotly |
| Data parsing | Pandas + regex |
| Schemas | Pydantic v2 |

## Constraints

- Parser node uses **zero LLM calls** — pure Python only
- Embeddings run **locally** via sentence-transformers (no API cost)
- Only **one LLM call** per full diagnosis run (Advisor node only)
- ChromaDB persists to `./chroma_db` directory
