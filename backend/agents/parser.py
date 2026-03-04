"""Node 1 — Parser ('The Intern').

Pure Python / pandas / regex. Zero LLM calls.
Accepts raw text artifacts and returns a structured SymptomSet.
"""
from __future__ import annotations

import io
import json
import math
import re
from typing import Any, Optional

import pandas as pd
import yaml

from backend.models import ConfigFlags, GraphState, LossEvent, SymptomSet

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_GPU_MEM_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*MB", re.IGNORECASE),
    re.compile(r"memory.*?:\s*(\d+(?:\.\d+)?)\s*MiB", re.IGNORECASE),
    re.compile(r"allocated.*?(\d+(?:\.\d+)?)\s*MB", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*MiB", re.IGNORECASE),
]

_LR_PATTERNS = [
    re.compile(r"lr\s*[=:]\s*([0-9eE+\-\.]+)", re.IGNORECASE),
    re.compile(r"learning.?rate\s*[=:]\s*([0-9eE+\-\.]+)", re.IGNORECASE),
]
_BS_PATTERNS = [
    re.compile(r"batch.?size\s*[=:]\s*(\d+)", re.IGNORECASE),
]
_OPT_PATTERNS = [
    re.compile(r"optimizer\s*[=:]\s*['\"]?(\w+)['\"]?", re.IGNORECASE),
]
_SCHED_PATTERNS = [
    re.compile(r"scheduler\s*[=:]\s*['\"]?(\w+)['\"]?", re.IGNORECASE),
    re.compile(r"lr.?scheduler\s*[=:]\s*['\"]?(\w+)['\"]?", re.IGNORECASE),
]
_GRAD_CLIP_PATTERNS = [
    re.compile(r"grad.?clip(?:_norm)?\s*[=:]\s*([0-9eE+\-\.]+)", re.IGNORECASE),
    re.compile(r"max.?norm\s*[=:]\s*([0-9eE+\-\.]+)", re.IGNORECASE),
    re.compile(r"clip.?grad\s*[=:]\s*([0-9eE+\-\.]+)", re.IGNORECASE),
]

# Stack-trace classification
_CUDA_OOM = re.compile(r"cuda\s+out\s+of\s+memory", re.IGNORECASE)
_CUDA_DEVICE = re.compile(
    r"expected\s+object\s+of\s+device\s+type\s+cuda.*got\s+device\s+type\s+cpu",
    re.IGNORECASE,
)
_CUDA_GENERAL = re.compile(r"(cuda|nccl)\s+error", re.IGNORECASE)
_SHAPE_MISMATCH = re.compile(
    r"(RuntimeError.*shape|size.*mismatch|mat.*cannot.*multipl|"
    r"expected.*batch_size.*match|index.*out.*of.*bounds.*embed)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_match(patterns: list[re.Pattern], text: str) -> Optional[str]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def _parse_csv(raw_csv: str) -> pd.DataFrame:
    """Try to parse CSV; accept various column naming conventions."""
    df = pd.read_csv(io.StringIO(raw_csv))

    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Map common aliases to canonical names
    step_aliases = {"step", "epoch", "iter", "iteration", "steps", "epochs"}
    loss_aliases = {"loss", "train_loss", "training_loss", "train loss", "value"}

    step_col = next((c for c in df.columns if c in step_aliases), None)
    loss_col = next((c for c in df.columns if c in loss_aliases), None)

    if step_col is None:
        # assume first column is step
        step_col = df.columns[0]
    if loss_col is None:
        # assume second column is loss
        loss_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    out = pd.DataFrame({"step": df[step_col], "loss": df[loss_col]})
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["loss"] = pd.to_numeric(out["loss"], errors="coerce")
    return out.dropna(subset=["step"]).reset_index(drop=True)


def _detect_events(df: pd.DataFrame) -> tuple[list[LossEvent], Optional[int]]:
    """Detect spikes, NaN rows, plateaus, and divergence step."""
    events: list[LossEvent] = []
    divergence_step: Optional[int] = None

    if df.empty:
        return events, divergence_step

    loss = df["loss"]
    steps = df["step"].astype(int)

    # NaN events
    nan_mask = loss.isna()
    for idx in df[nan_mask].index:
        events.append(
            LossEvent(
                event_type="nan",
                step=int(steps[idx]),
                description="Loss value is NaN",
            )
        )

    # Work with non-NaN rows for further analysis
    clean = df.dropna(subset=["loss"]).copy()
    if clean.empty:
        return events, divergence_step

    # Divergence / Inf
    inf_mask = clean["loss"].apply(lambda x: math.isinf(x) if isinstance(x, float) else False)
    if inf_mask.any():
        div_idx = clean[inf_mask].index[0]
        divergence_step = int(clean.loc[div_idx, "step"])
        events.append(
            LossEvent(
                event_type="divergence",
                step=divergence_step,
                value=clean.loc[div_idx, "loss"],
                description="Loss diverged to Inf",
            )
        )

    # Spike detection: loss > rolling_mean + 4σ (z-score based, scale-invariant).
    # This naturally adapts whether loss is 2.3 (early training) or 0.05 (converged),
    # avoiding the false positives of a fixed multiplier threshold.
    if len(clean) >= 5:
        rolling_mean = clean["loss"].rolling(window=5, min_periods=1).mean().shift(1)
        rolling_std = clean["loss"].rolling(window=5, min_periods=2).std().shift(1)
        spike_mask = clean["loss"] > (rolling_mean + 4 * rolling_std)
        # Suppress the first few steps before std estimate is stable
        spike_mask.iloc[:4] = False
        for idx in clean[spike_mask].index:
            step_val = int(clean.loc[idx, "step"])
            loss_val = float(clean.loc[idx, "loss"])
            events.append(
                LossEvent(
                    event_type="spike",
                    step=step_val,
                    value=loss_val,
                    description=f"Loss spike detected (value={loss_val:.4f})",
                )
            )
            if divergence_step is None:
                divergence_step = step_val

    # Plateau detection: rolling std < 0.001 over 10 consecutive steps
    if len(clean) >= 10:
        rolling_std = clean["loss"].rolling(window=10).std()
        plateau_mask = rolling_std < 0.001
        # find first plateau start
        if plateau_mask.any():
            first_plateau_idx = plateau_mask.idxmax()
            plateau_step = int(clean.loc[first_plateau_idx, "step"])
            events.append(
                LossEvent(
                    event_type="plateau",
                    step=plateau_step,
                    description=f"Loss plateau detected starting at step {plateau_step}",
                )
            )

    return events, divergence_step


def _convergence_speed(df: pd.DataFrame) -> Optional[str]:
    """Classify convergence as fast/slow/normal from first 5 epoch-worth of data."""
    if df.empty or len(df) < 2:
        return None

    clean = df.dropna(subset=["loss"])
    # Use first 20% of data or first 5 rows, whichever is larger
    n = max(5, len(clean) // 5)
    subset = clean.head(n)

    if len(subset) < 2:
        return None

    x = subset["step"].values.astype(float)
    y = subset["loss"].values.astype(float)

    # Linear regression slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return None
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom

    if slope < -0.05:
        return "fast"
    elif slope > -0.01:
        return "slow"
    return "normal"


def _extract_gpu_memory(log_text: str) -> Optional[float]:
    max_mb: Optional[float] = None
    for pat in _GPU_MEM_PATTERNS:
        for m in pat.finditer(log_text):
            try:
                val = float(m.group(1))
                if max_mb is None or val > max_mb:
                    max_mb = val
            except ValueError:
                pass
    return max_mb


def _classify_error(text: str) -> tuple[Optional[str], Optional[str]]:
    """Return (error_type, offending_line) from a stack trace string."""
    if not text:
        return None, None

    if _CUDA_OOM.search(text):
        error_type = "OOM"
    elif _CUDA_DEVICE.search(text):
        error_type = "DeviceMismatch"
    elif _CUDA_GENERAL.search(text):
        error_type = "CUDA"
    elif _SHAPE_MISMATCH.search(text):
        error_type = "ShapeMismatch"
    else:
        error_type = "Other"

    # Extract the final RuntimeError / Error line as offending line
    error_line: Optional[str] = None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line and ("Error" in line or "error" in line):
            error_line = line
            break
    if error_line is None and text.strip():
        # fallback: last non-empty line
        for line in reversed(text.splitlines()):
            if line.strip():
                error_line = line.strip()
                break

    return error_type, error_line


def _extract_config_flags(raw_config: Optional[str]) -> ConfigFlags:
    if not raw_config:
        return ConfigFlags()

    # Try YAML first, then JSON
    cfg: dict[str, Any] = {}
    try:
        cfg = yaml.safe_load(raw_config) or {}
    except Exception:
        try:
            cfg = json.loads(raw_config)
        except Exception:
            pass

    def _deep_get(d: Any, *keys: str) -> Optional[Any]:
        """Search nested dict for a key."""
        if isinstance(d, dict):
            for k, v in d.items():
                if k.lower() in keys:
                    return v
                result = _deep_get(v, *keys)
                if result is not None:
                    return result
        return None

    lr = _deep_get(cfg, "lr", "learning_rate")
    bs = _deep_get(cfg, "batch_size", "batchsize", "batch")
    opt = _deep_get(cfg, "optimizer", "optim")
    sched = _deep_get(cfg, "scheduler", "lr_scheduler")
    grad_clip = _deep_get(cfg, "grad_clip", "gradient_clip", "max_norm", "clip_grad_norm")

    def _safe_float(v: Any) -> Optional[float]:
        try:
            return float(v)
        except Exception:
            return None

    def _safe_int(v: Any) -> Optional[int]:
        try:
            return int(v)
        except Exception:
            return None

    return ConfigFlags(
        learning_rate=_safe_float(lr),
        batch_size=_safe_int(bs),
        optimizer=str(opt) if opt is not None else None,
        scheduler=str(sched) if sched is not None else None,
        grad_clip=_safe_float(grad_clip),
    )


# ---------------------------------------------------------------------------
# Main node function
# ---------------------------------------------------------------------------

def parse_node(state: GraphState) -> GraphState:
    """LangGraph node: parse raw inputs into a SymptomSet."""
    raw_log: Optional[str] = state.get("raw_log")
    raw_csv: Optional[str] = state.get("raw_csv")
    raw_config: Optional[str] = state.get("raw_config")
    stack_trace: Optional[str] = state.get("stack_trace")

    # Combine all text sources for GPU memory / config scanning
    all_text = " ".join(filter(None, [raw_log, raw_config, stack_trace]))

    # --- Loss trajectory ---
    loss_trajectory: list[tuple[int, float]] = []
    loss_events: list[LossEvent] = []
    divergence_step: Optional[int] = None
    convergence_speed: Optional[str] = None

    if raw_csv:
        try:
            df = _parse_csv(raw_csv)
            loss_trajectory = [
                (int(row.step), float(row.loss))
                for row in df.itertuples()
                if not (isinstance(row.loss, float) and math.isnan(row.loss))
            ]
            loss_events, divergence_step = _detect_events(df)
            convergence_speed = _convergence_speed(df)
        except Exception as e:
            loss_events.append(
                LossEvent(event_type="other", description=f"CSV parse error: {e}")
            )

    # --- GPU memory ---
    gpu_memory_usage = _extract_gpu_memory(all_text) if all_text else None

    # --- Error classification ---
    # Prefer dedicated stack_trace field; fall back to raw_log
    trace_text = stack_trace or raw_log or ""
    error_type, error_line = _classify_error(trace_text)

    # --- Config flags ---
    config_flags = _extract_config_flags(raw_config)

    # If no YAML/JSON config, try extracting from log text
    if raw_config is None and all_text:
        inline_flags = ConfigFlags(
            learning_rate=_safe_float_match(_LR_PATTERNS, all_text),
            batch_size=_safe_int_match(_BS_PATTERNS, all_text),
            optimizer=_first_match(_OPT_PATTERNS, all_text),
            scheduler=_first_match(_SCHED_PATTERNS, all_text),
            grad_clip=_safe_float_match(_GRAD_CLIP_PATTERNS, all_text),
        )
        # Merge: config file wins, inline fills gaps
        config_flags = ConfigFlags(
            learning_rate=config_flags.learning_rate or inline_flags.learning_rate,
            batch_size=config_flags.batch_size or inline_flags.batch_size,
            optimizer=config_flags.optimizer or inline_flags.optimizer,
            scheduler=config_flags.scheduler or inline_flags.scheduler,
            grad_clip=config_flags.grad_clip or inline_flags.grad_clip,
        )

    symptom_set = SymptomSet(
        loss_trajectory=loss_trajectory,
        loss_events=loss_events,
        gpu_memory_usage=gpu_memory_usage,
        error_type=error_type,
        error_line=error_line,
        config_flags=config_flags,
        convergence_speed=convergence_speed,
        raw_log=raw_log,
        raw_config=raw_config,
    )

    return {**state, "symptom_set": symptom_set}


# ---------------------------------------------------------------------------
# Tiny helpers used in fallback inline extraction
# ---------------------------------------------------------------------------

def _safe_float_match(patterns: list[re.Pattern], text: str) -> Optional[float]:
    val = _first_match(patterns, text)
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _safe_int_match(patterns: list[re.Pattern], text: str) -> Optional[int]:
    val = _first_match(patterns, text)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None
