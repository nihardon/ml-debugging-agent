"""FastAPI application exposing /diagnose and /health endpoints."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Load .env from project root regardless of where uvicorn is invoked
load_dotenv(Path(__file__).parents[1] / ".env")

from backend.graph import graph
from backend.kb.chroma_store import get_store
from backend.models import DiagnosticReport

app = FastAPI(title="ML Debugging Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        store = get_store()
        doc_count = store.count()
        seeded = doc_count > 0
    except Exception as exc:
        return {"status": "error", "kb_doc_count": 0, "error": str(exc)}

    return {
        "status": "ok",
        "kb_doc_count": doc_count,
        "kb_seeded": seeded,
    }


# ---------------------------------------------------------------------------
# /diagnose
# ---------------------------------------------------------------------------

@app.post("/diagnose", response_model=DiagnosticReport)
async def diagnose(
    log_file: Optional[UploadFile] = File(default=None, description=".txt training log"),
    csv_file: Optional[UploadFile] = File(default=None, description=".csv loss curve"),
    config_file: Optional[UploadFile] = File(default=None, description=".yaml or .json config"),
    stack_trace: Optional[str] = Form(default=None, description="Pasted stack trace text"),
):
    # Reject if absolutely nothing was provided
    if not any([log_file, csv_file, config_file, stack_trace]):
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: log_file, csv_file, config_file, stack_trace.",
        )

    # Read uploaded file contents
    raw_log: Optional[str] = None
    raw_csv: Optional[str] = None
    raw_config: Optional[str] = None

    if log_file is not None:
        content = await log_file.read()
        raw_log = content.decode("utf-8", errors="replace")

    if csv_file is not None:
        content = await csv_file.read()
        raw_csv = content.decode("utf-8", errors="replace")

    if config_file is not None:
        content = await config_file.read()
        raw_config = content.decode("utf-8", errors="replace")

    # Build initial LangGraph state
    initial_state = {
        "raw_log": raw_log,
        "raw_csv": raw_csv,
        "raw_config": raw_config,
        "stack_trace": stack_trace,
        "symptom_set": None,
        "retrieved_docs": None,
        "diagnostic_report": None,
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    report: Optional[DiagnosticReport] = final_state.get("diagnostic_report")
    if report is None:
        raise HTTPException(status_code=500, detail="Agent did not produce a diagnostic report.")

    return report
