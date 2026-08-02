"""Tests for selected model-effort confirmation execution."""

from __future__ import annotations

import coding_model_router_model_effort_confirm as confirm
import coding_model_router_swerebench_execute as runner


def test_configure_binds_exact_single_arm() -> None:
    originals = {
        "model": runner.MODEL,
        "efforts": runner.EFFORTS,
        "phase": runner.DEVELOPMENT_PHASE,
        "validator": runner.REMOTE_VALIDATOR,
        "reused": runner.REUSED_TASKS,
        "archives": runner.SMOKE_ARCHIVE_SHA256,
        "spend": runner.DEFAULT_PRIOR_SPEND_USD,
        "authorization": runner.EXTERNAL_AUTHORIZATION,
    }
    try:
        _, arm = confirm.configure("gpt-5.6-sol", "xhigh", 2_000.0)
        assert arm == "sol-xhigh"
        assert runner.MODEL == "gpt-5.6-sol"
        assert runner.EFFORTS == ("xhigh",)
        assert runner.DEVELOPMENT_PHASE.corpus_sha256 == confirm.CONFIRMATION_CORPUS_SHA256
        assert runner.DEVELOPMENT_PHASE.reuse_smoke is False
        assert runner.DEFAULT_PRIOR_SPEND_USD == 2_000.0
    finally:
        runner.MODEL = originals["model"]
        runner.EFFORTS = originals["efforts"]
        runner.DEVELOPMENT_PHASE = originals["phase"]
        runner.REMOTE_VALIDATOR = originals["validator"]
        runner.REUSED_TASKS = originals["reused"]
        runner.SMOKE_ARCHIVE_SHA256 = originals["archives"]
        runner.DEFAULT_PRIOR_SPEND_USD = originals["spend"]
        runner.EXTERNAL_AUTHORIZATION = originals["authorization"]
