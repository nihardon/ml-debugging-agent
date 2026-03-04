"""Node 3 — Advisor ('The Doctor').

Makes a single async call to Claude and parses the structured DiagnosticReport.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import anthropic

from backend.models import (
    Citation,
    DiagnosticReport,
    GraphState,
    KBDocument,
    RankedAction,
    SymptomSet,
)

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a senior ML engineer and debugging expert. \
You will be given a structured set of symptoms extracted from a user's ML training run \
(loss curves, stack traces, config) alongside relevant knowledge-base documents. \

Your job is to diagnose the root cause and return a JSON object that strictly follows \
this schema (no extra keys, no markdown, just raw JSON):

{
  "status": "Critical" | "Warning" | "Healthy",
  "root_cause": "<one concise sentence>",
  "confidence": <float 0.0-1.0>,
  "explanation": "<2-3 sentences in plain English>",
  "fix_code_snippet": "<actual runnable PyTorch/Python code>",
  "ranked_actions": [
    {"action": "<what to do>", "priority": 1},
    {"action": "<what to do>", "priority": 2},
    {"action": "<what to do>", "priority": 3}
  ],
  "what_to_monitor": "<metric or signal to watch after applying the fix>",
  "citations": [
    {"id": "<doc_id from KB>", "source": "<citation string>"}
  ],
  "divergence_step": <integer or null>
}

Rules:
- status is "Critical" if the run crashed or loss diverged/NaN, "Warning" if there are \
  concerning signs but training continued, "Healthy" if nothing alarming was detected.
- confidence reflects how certain you are given the evidence (0.0 = no idea, 1.0 = certain).
- ranked_actions should have priority 1 = most urgent.
- citations must reference the KB documents provided; use their doc_id as the id field.
- divergence_step is the training step where loss first diverged/spiked, or null.
- Return ONLY the JSON object, no preamble or explanation outside it.
"""


def _build_user_message(symptoms: SymptomSet, docs: list[KBDocument]) -> str:
    symptom_dict = symptoms.model_dump(exclude={"raw_log", "raw_config"})
    docs_list = [d.model_dump() for d in docs]

    return json.dumps(
        {
            "symptoms": symptom_dict,
            "knowledge_base_matches": docs_list,
        },
        indent=2,
        default=str,
    )


def _extract_json(text: str) -> str:
    """Strip any surrounding markdown code fences if the model adds them."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
    if fence:
        return fence.group(1).strip()
    return text


async def advise_node(state: GraphState) -> GraphState:
    """LangGraph node: call Claude once and parse a DiagnosticReport."""
    symptoms: Optional[SymptomSet] = state.get("symptom_set")
    docs: list[KBDocument] = state.get("retrieved_docs") or []

    if symptoms is None:
        report = DiagnosticReport(
            status="Healthy",
            root_cause="No training artifacts were provided.",
            confidence=0.0,
            explanation="Upload a CSV loss curve, log file, or stack trace to get a diagnosis.",
            fix_code_snippet="# No artifacts to analyze",
            ranked_actions=[],
            what_to_monitor="N/A",
            citations=[],
            divergence_step=None,
        )
        return {**state, "diagnostic_report": report}

    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    user_message = _build_user_message(symptoms, docs)

    response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_json = _extract_json(response.content[0].text)

    try:
        report = DiagnosticReport.model_validate_json(raw_json)
    except Exception:
        # Fallback: try parsing manually and reconstructing
        parsed = json.loads(raw_json)
        parsed["ranked_actions"] = [
            RankedAction(**a) if isinstance(a, dict) else a
            for a in parsed.get("ranked_actions", [])
        ]
        parsed["citations"] = [
            Citation(**c) if isinstance(c, dict) else c
            for c in parsed.get("citations", [])
        ]
        report = DiagnosticReport(**parsed)

    return {**state, "diagnostic_report": report}
