"""Unit tests for backend/agents/parser.py.

Run from the project root:
    pytest tests/test_parser.py -v
"""
from __future__ import annotations

import math
import textwrap

import pandas as pd
import pytest

from backend.agents.parser import (
    _classify_error,
    _convergence_speed,
    _detect_events,
    _extract_config_flags,
    _extract_gpu_memory,
    _parse_csv,
    parse_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(csv=None, log=None, config=None, stack_trace=None):
    return {
        "raw_csv": csv,
        "raw_log": log,
        "raw_config": config,
        "stack_trace": stack_trace,
        "symptom_set": None,
        "retrieved_docs": None,
        "diagnostic_report": None,
    }


def event_types(events):
    return [e.event_type for e in events]


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

class TestParseCsv:
    def test_standard_columns(self):
        csv = "step,loss\n0,2.3\n1,2.1\n2,1.9\n"
        df = _parse_csv(csv)
        assert list(df.columns) == ["step", "loss"]
        assert len(df) == 3
        assert df["step"].tolist() == [0, 1, 2]

    def test_epoch_alias(self):
        csv = "epoch,train_loss\n1,2.3\n2,2.0\n3,1.7\n"
        df = _parse_csv(csv)
        assert df["step"].tolist() == [1, 2, 3]
        assert pytest.approx(df["loss"].tolist()) == [2.3, 2.0, 1.7]

    def test_windows_line_endings(self):
        csv = "step,loss\r\n0,2.3\r\n1,2.1\r\n"
        df = _parse_csv(csv)
        assert len(df) == 2
        assert not df["loss"].isna().any()

    def test_fallback_to_column_index(self):
        csv = "x,y\n0,1.5\n1,1.2\n"
        df = _parse_csv(csv)
        assert df["step"].tolist() == [0, 1]
        assert pytest.approx(df["loss"].tolist()) == [1.5, 1.2]

    def test_non_numeric_rows_dropped(self):
        csv = "step,loss\n0,2.3\nnote,bad\n2,1.9\n"
        df = _parse_csv(csv)
        # 'note' row has non-numeric step → coerced to NaN → dropped
        assert len(df) == 2

    def test_uppercase_headers(self):
        csv = "Step,Loss\n0,2.3\n1,2.0\n"
        df = _parse_csv(csv)
        assert "step" in df.columns
        assert "loss" in df.columns


# ---------------------------------------------------------------------------
# Event detection — NaN
# ---------------------------------------------------------------------------

class TestNanDetection:
    def test_nan_row_flagged(self):
        csv = "step,loss\n0,2.3\n1,\n2,1.9\n"
        df = _parse_csv(csv)
        events, _ = _detect_events(df)
        assert "nan" in event_types(events)
        nan_event = next(e for e in events if e.event_type == "nan")
        assert nan_event.step == 1

    def test_no_nan_no_event(self):
        csv = "step,loss\n0,2.3\n1,2.1\n2,1.9\n"
        df = _parse_csv(csv)
        events, _ = _detect_events(df)
        assert "nan" not in event_types(events)


# ---------------------------------------------------------------------------
# Event detection — Divergence (Inf)
# ---------------------------------------------------------------------------

class TestDivergenceDetection:
    def test_inf_triggers_divergence(self):
        csv = "step,loss\n0,2.3\n1,2.1\n2,inf\n3,inf\n"
        df = _parse_csv(csv)
        events, div_step = _detect_events(df)
        assert "divergence" in event_types(events)
        assert div_step == 2

    def test_divergence_step_is_first_inf(self):
        csv = "step,loss\n0,2.3\n5,2.1\n10,inf\n15,inf\n"
        df = _parse_csv(csv)
        _, div_step = _detect_events(df)
        assert div_step == 10

    def test_healthy_loss_no_divergence(self):
        csv = "step,loss\n" + "\n".join(f"{i},{2.3 - i*0.1}" for i in range(10))
        df = _parse_csv(csv)
        events, div_step = _detect_events(df)
        assert "divergence" not in event_types(events)
        assert div_step is None


# ---------------------------------------------------------------------------
# Event detection — Spike (z-score)
# ---------------------------------------------------------------------------

class TestSpikeDetection:
    def _make_spike_csv(self, base=0.5, spike_val=5.0, spike_step=8):
        """Smooth decreasing loss with a single spike."""
        rows = ["step,loss"]
        for i in range(15):
            val = spike_val if i == spike_step else max(0.05, base - i * 0.03)
            rows.append(f"{i},{val}")
        return "\n".join(rows)

    def test_genuine_spike_detected(self):
        csv = self._make_spike_csv(base=0.5, spike_val=8.0, spike_step=8)
        df = _parse_csv(csv)
        events, _ = _detect_events(df)
        assert "spike" in event_types(events)

    def test_spike_step_recorded(self):
        csv = self._make_spike_csv(base=0.5, spike_val=8.0, spike_step=8)
        df = _parse_csv(csv)
        events, div_step = _detect_events(df)
        spike_event = next(e for e in events if e.event_type == "spike")
        assert spike_event.step == 8
        assert div_step == 8

    def test_no_false_positive_normal_decrease(self):
        """Smooth monotonically decreasing loss must not trigger spike."""
        rows = ["step,loss"]
        for i in range(20):
            rows.append(f"{i},{2.3 - i * 0.1}")
        df = _parse_csv("\n".join(rows))
        events, _ = _detect_events(df)
        assert "spike" not in event_types(events)

    def test_no_false_positive_converged_noise(self):
        """Small noisy fluctuations at converged loss must not be flagged.

        This is the regression test for the 2x→4σ threshold change.
        Old threshold (2x rolling mean) would have flagged ~0.22 when mean=0.15.
        New z-score threshold should not.
        """
        import random
        random.seed(42)
        rows = ["step,loss"]
        for i in range(30):
            # Loss converged to ~0.15 with ±0.03 noise
            val = 0.15 + random.uniform(-0.03, 0.03)
            rows.append(f"{i},{val:.4f}")
        df = _parse_csv("\n".join(rows))
        events, _ = _detect_events(df)
        assert "spike" not in event_types(events), (
            "False spike detected on noisy-but-converged loss — "
            "z-score threshold regression"
        )

    def test_no_spike_with_fewer_than_5_rows(self):
        csv = "step,loss\n0,2.3\n1,2.1\n2,1.9\n"
        df = _parse_csv(csv)
        events, _ = _detect_events(df)
        assert "spike" not in event_types(events)


# ---------------------------------------------------------------------------
# Event detection — Plateau
# ---------------------------------------------------------------------------

class TestPlateauDetection:
    def test_plateau_detected(self):
        rows = ["step,loss"]
        for i in range(5):
            rows.append(f"{i},{2.3 - i * 0.2}")
        # perfectly flat for 10 steps
        for i in range(5, 15):
            rows.append(f"{i},0.1000")
        df = _parse_csv("\n".join(rows))
        events, _ = _detect_events(df)
        assert "plateau" in event_types(events)

    def test_no_plateau_on_short_series(self):
        csv = "step,loss\n0,2.3\n1,2.1\n2,1.9\n"
        df = _parse_csv(csv)
        events, _ = _detect_events(df)
        assert "plateau" not in event_types(events)


# ---------------------------------------------------------------------------
# Convergence speed
# ---------------------------------------------------------------------------

class TestConvergenceSpeed:
    def _df(self, pairs):
        return pd.DataFrame({"step": [p[0] for p in pairs], "loss": [p[1] for p in pairs]})

    def test_fast(self):
        # steep drop: slope < -0.05
        df = self._df([(i, 2.3 - i * 0.3) for i in range(10)])
        assert _convergence_speed(df) == "fast"

    def test_slow(self):
        # shallow drop: slope > -0.01
        df = self._df([(i, 2.3 - i * 0.001) for i in range(10)])
        assert _convergence_speed(df) == "slow"

    def test_normal(self):
        # moderate: -0.05 <= slope <= -0.01
        df = self._df([(i, 2.3 - i * 0.02) for i in range(10)])
        assert _convergence_speed(df) == "normal"

    def test_returns_none_on_single_row(self):
        df = self._df([(0, 2.3)])
        assert _convergence_speed(df) is None

    def test_returns_none_on_empty(self):
        df = pd.DataFrame({"step": [], "loss": []})
        assert _convergence_speed(df) is None


# ---------------------------------------------------------------------------
# GPU memory extraction
# ---------------------------------------------------------------------------

class TestGpuMemory:
    def test_mb_suffix(self):
        assert _extract_gpu_memory("Allocated 4096 MB on device") == 4096.0

    def test_mib_suffix(self):
        assert _extract_gpu_memory("memory: 8192 MiB") == 8192.0

    def test_picks_max(self):
        assert _extract_gpu_memory("Step 1: 512 MB, Step 2: 2048 MB") == 2048.0

    def test_no_match_returns_none(self):
        assert _extract_gpu_memory("Training started, epoch 1") is None

    def test_empty_string(self):
        assert _extract_gpu_memory("") is None


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestClassifyError:
    def test_oom(self):
        trace = "RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB"
        error_type, error_line = _classify_error(trace)
        assert error_type == "OOM"
        assert "memory" in error_line.lower()

    def test_device_mismatch(self):
        trace = "RuntimeError: Expected object of device type cuda but got device type cpu"
        error_type, _ = _classify_error(trace)
        assert error_type == "DeviceMismatch"

    def test_shape_mismatch_mat_mul(self):
        trace = "RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x512 and 256x10)"
        error_type, _ = _classify_error(trace)
        assert error_type == "ShapeMismatch"

    def test_shape_mismatch_size(self):
        trace = "RuntimeError: size mismatch, m1: [32 x 512], m2: [256 x 10]"
        error_type, _ = _classify_error(trace)
        assert error_type == "ShapeMismatch"

    def test_nccl_cuda_error(self):
        trace = "RuntimeError: NCCL error in: ...\nncclSystemError: CUDA error"
        error_type, _ = _classify_error(trace)
        assert error_type in ("CUDA", "OOM")  # CUDA error general

    def test_other(self):
        trace = "ValueError: Expected positive integer for n_clusters"
        error_type, _ = _classify_error(trace)
        assert error_type == "Other"

    def test_empty_returns_none(self):
        error_type, error_line = _classify_error("")
        assert error_type is None
        assert error_line is None

    def test_error_line_extracted(self):
        trace = textwrap.dedent("""\
            Traceback (most recent call last):
              File "train.py", line 42, in forward
                output = self.linear(x)
            RuntimeError: mat1 and mat2 shapes cannot be multiplied
        """)
        _, error_line = _classify_error(trace)
        assert "RuntimeError" in error_line


# ---------------------------------------------------------------------------
# Config flag extraction
# ---------------------------------------------------------------------------

class TestExtractConfigFlags:
    def test_yaml_flat(self):
        yaml_str = "lr: 0.001\nbatch_size: 32\noptimizer: Adam\n"
        flags = _extract_config_flags(yaml_str)
        assert flags.learning_rate == pytest.approx(0.001)
        assert flags.batch_size == 32
        assert flags.optimizer == "Adam"

    def test_yaml_nested(self):
        yaml_str = textwrap.dedent("""\
            training:
              lr: 3e-4
              batch_size: 64
              grad_clip: 1.0
        """)
        flags = _extract_config_flags(yaml_str)
        assert flags.learning_rate == pytest.approx(3e-4)
        assert flags.batch_size == 64
        assert flags.grad_clip == pytest.approx(1.0)

    def test_json(self):
        import json
        cfg = json.dumps({"learning_rate": 1e-3, "batch_size": 16, "optimizer": "AdamW"})
        flags = _extract_config_flags(cfg)
        assert flags.learning_rate == pytest.approx(1e-3)
        assert flags.batch_size == 16
        assert flags.optimizer == "AdamW"

    def test_missing_fields_are_none(self):
        flags = _extract_config_flags("lr: 0.01\n")
        assert flags.batch_size is None
        assert flags.optimizer is None
        assert flags.scheduler is None
        assert flags.grad_clip is None

    def test_none_returns_empty_flags(self):
        flags = _extract_config_flags(None)
        assert flags.learning_rate is None
        assert flags.batch_size is None


# ---------------------------------------------------------------------------
# Full parse_node integration
# ---------------------------------------------------------------------------

class TestParseNode:
    def test_csv_only_populates_trajectory(self):
        csv = "step,loss\n0,2.3\n1,2.1\n2,1.9\n"
        state = parse_node(make_state(csv=csv))
        ss = state["symptom_set"]
        assert len(ss.loss_trajectory) == 3
        assert ss.loss_trajectory[0] == (0, 2.3)

    def test_stack_trace_sets_error_type(self):
        state = parse_node(make_state(
            stack_trace="RuntimeError: CUDA out of memory. Tried to allocate 4 GiB"
        ))
        ss = state["symptom_set"]
        assert ss.error_type == "OOM"
        assert ss.error_line is not None

    def test_log_fallback_for_error(self):
        """stack_trace field absent — should fall back to raw_log."""
        state = parse_node(make_state(
            log="RuntimeError: CUDA out of memory."
        ))
        ss = state["symptom_set"]
        assert ss.error_type == "OOM"

    def test_config_yaml_parsed(self):
        state = parse_node(make_state(config="lr: 0.001\nbatch_size: 64\n"))
        ss = state["symptom_set"]
        assert ss.config_flags.learning_rate == pytest.approx(0.001)
        assert ss.config_flags.batch_size == 64

    def test_raw_config_stored_as_string(self):
        """Regression: raw_config in SymptomSet must be str, not dict."""
        raw = "lr: 0.001\nbatch_size: 64\n"
        state = parse_node(make_state(config=raw))
        ss = state["symptom_set"]
        assert isinstance(ss.raw_config, str)
        assert ss.raw_config == raw

    def test_inline_log_config_extraction(self):
        """Config flags extracted from log text when no config file provided."""
        log = "Training start | lr=0.001 | batch_size=32 | optimizer=SGD"
        state = parse_node(make_state(log=log))
        ss = state["symptom_set"]
        assert ss.config_flags.learning_rate == pytest.approx(0.001)
        assert ss.config_flags.batch_size == 32
        assert ss.config_flags.optimizer == "SGD"

    def test_gpu_memory_from_log(self):
        log = "GPU memory allocated: 6144 MB"
        state = parse_node(make_state(log=log))
        ss = state["symptom_set"]
        assert ss.gpu_memory_usage == pytest.approx(6144.0)

    def test_divergence_step_set_on_inf(self):
        csv = "step,loss\n0,2.3\n1,2.1\n2,inf\n"
        state = parse_node(make_state(csv=csv))
        ss = state["symptom_set"]
        divergence_events = [e for e in ss.loss_events if e.event_type == "divergence"]
        assert len(divergence_events) == 1
        assert divergence_events[0].step == 2

    def test_empty_state_produces_empty_symptom_set(self):
        """parse_node must not crash on completely empty input."""
        state = parse_node(make_state())
        ss = state["symptom_set"]
        assert ss is not None
        assert ss.loss_trajectory == []
        assert ss.error_type is None
        assert ss.gpu_memory_usage is None

    def test_convergence_speed_fast(self):
        rows = ["step,loss"] + [f"{i},{2.3 - i * 0.3}" for i in range(10)]
        state = parse_node(make_state(csv="\n".join(rows)))
        assert state["symptom_set"].convergence_speed == "fast"

    def test_convergence_speed_slow(self):
        rows = ["step,loss"] + [f"{i},{2.3 - i * 0.001}" for i in range(10)]
        state = parse_node(make_state(csv="\n".join(rows)))
        assert state["symptom_set"].convergence_speed == "slow"
