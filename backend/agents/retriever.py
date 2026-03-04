"""Node 2 — Retriever ('The Librarian').

Builds a query string from the top 3 most salient symptoms in a SymptomSet
and retrieves the top-5 matching documents from ChromaDB.
"""
from __future__ import annotations

from typing import Optional

from backend.kb.chroma_store import get_store
from backend.models import GraphState, KBDocument, SymptomSet


def _build_query(symptoms: SymptomSet) -> str:
    """Compose a query string from the most salient symptom signals."""
    parts: list[str] = []

    # 1. Error type (highest signal)
    if symptoms.error_type and symptoms.error_type != "Other":
        if symptoms.error_type == "OOM":
            parts.append("CUDA out of memory")
        elif symptoms.error_type == "DeviceMismatch":
            parts.append("tensor device mismatch cuda cpu")
        elif symptoms.error_type == "ShapeMismatch":
            parts.append("shape mismatch RuntimeError tensor dimensions")
        elif symptoms.error_type == "CUDA":
            parts.append("CUDA error distributed training")
        else:
            parts.append(symptoms.error_type)

    if symptoms.error_line:
        parts.append(symptoms.error_line[:200])

    # 2. Loss events (second-highest signal)
    event_types = {e.event_type for e in symptoms.loss_events}
    if "divergence" in event_types:
        parts.append("loss diverged Inf NaN exploding gradients")
    if "nan" in event_types:
        parts.append("loss NaN numerical instability")
    if "spike" in event_types:
        parts.append("loss spike instability learning rate too high")
    if "plateau" in event_types:
        parts.append("loss plateau not converging")

    # 3. Convergence speed
    if symptoms.convergence_speed == "slow":
        parts.append("loss decreasing slowly learning rate too low")
    elif symptoms.convergence_speed == "fast" and not event_types:
        parts.append("fast convergence potential overfitting")

    # 4. Config signals
    cfg = symptoms.config_flags
    if cfg.learning_rate is not None:
        if cfg.learning_rate > 0.1:
            parts.append("learning rate too high instability")
        elif cfg.learning_rate < 1e-5:
            parts.append("learning rate too low slow convergence")
    if cfg.grad_clip is None and ("spike" in event_types or "divergence" in event_types):
        parts.append("missing gradient clipping exploding gradients")

    # Fallback
    if not parts:
        parts.append("training loss not converging machine learning debugging")

    return " ".join(parts[:3])  # cap to top 3 most salient parts


def retrieve_node(state: GraphState) -> GraphState:
    """LangGraph node: retrieve relevant KB documents for the symptom set."""
    symptoms: Optional[SymptomSet] = state.get("symptom_set")

    if symptoms is None:
        return {**state, "retrieved_docs": []}

    query_text = _build_query(symptoms)

    store = get_store()

    if store.count() == 0:
        return {**state, "retrieved_docs": []}

    raw_docs = store.query(query_text, n_results=5)

    retrieved: list[KBDocument] = [
        KBDocument(
            doc_id=d["doc_id"],
            symptom=d["symptom"],
            diagnosis=d["diagnosis"],
            fix=d["fix"],
            code_snippet=d.get("code_snippet", ""),
            citation=d["citation"],
            domain=d["domain"],
        )
        for d in raw_docs
    ]

    return {**state, "retrieved_docs": retrieved}
