from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Parser output
# ---------------------------------------------------------------------------

class LossEvent(BaseModel):
    event_type: str  # "spike" | "nan" | "plateau" | "divergence"
    step: Optional[int] = None
    value: Optional[float] = None
    description: str = ""


class ConfigFlags(BaseModel):
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    optimizer: Optional[str] = None
    scheduler: Optional[str] = None
    grad_clip: Optional[float] = None


class SymptomSet(BaseModel):
    loss_trajectory: list[tuple[int, float]] = Field(default_factory=list)
    val_loss_trajectory: list[tuple[int, float]] = Field(default_factory=list)
    loss_events: list[LossEvent] = Field(default_factory=list)
    gpu_memory_usage: Optional[float] = None  # peak MB
    error_type: Optional[str] = None  # CUDA | OOM | ShapeMismatch | DeviceMismatch | Other
    error_line: Optional[str] = None
    config_flags: ConfigFlags = Field(default_factory=ConfigFlags)
    convergence_speed: Optional[str] = None  # "fast" | "slow" | "normal"
    raw_log: Optional[str] = None
    raw_config: Optional[str] = None  # raw config file text (YAML or JSON)


# ---------------------------------------------------------------------------
# Retriever output
# ---------------------------------------------------------------------------

class KBDocument(BaseModel):
    doc_id: str
    symptom: str
    diagnosis: str
    fix: str
    code_snippet: str = ""
    citation: str
    domain: str


# ---------------------------------------------------------------------------
# Advisor output (DiagnosticReport)
# ---------------------------------------------------------------------------

class RankedAction(BaseModel):
    action: str
    priority: int  # 1 | 2 | 3


class Citation(BaseModel):
    id: str
    source: str


class DiagnosticReport(BaseModel):
    status: str  # "Critical" | "Warning" | "Healthy"
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    fix_code_snippet: str
    ranked_actions: list[RankedAction] = Field(default_factory=list)
    what_to_monitor: str
    citations: list[Citation] = Field(default_factory=list)
    divergence_step: Optional[int] = None


# ---------------------------------------------------------------------------
# LangGraph state — carries data between nodes
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    # Inputs from API
    raw_log: Optional[str]
    raw_csv: Optional[str]
    raw_config: Optional[str]
    stack_trace: Optional[str]

    # Node outputs
    symptom_set: Optional[SymptomSet]
    retrieved_docs: Optional[list[KBDocument]]
    diagnostic_report: Optional[DiagnosticReport]
