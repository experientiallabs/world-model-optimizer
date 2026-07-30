"""Rung A analysis: pick per-corpus thresholds by the pre-registered rule, report holdout.

Registered rule (findings/c1.md round 2 section, written before the sweep ran): calibrate
on the fit half only; the calibrated threshold is the one whose FIT keep ratio lands
nearest the middle of the target band (keep 0.60-0.70, the band where the live grid held
cost flat); report everything on the holdout half. The calibrated threshold is a fit
OUTPUT and is recorded in result, never in params (dashboard convention).

Also validates the live anchor: the financebench curve at threshold 0.5 should sit near
the running grid's live aggregate keep (0.6554 when read on 2026-07-27); a large gap
would mean the trace-observation segments misrepresent live structure.

Usage: python3 .agents/scripts/analyze_threshold_sweep.py [sweep-results.json]
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("analyze_sweep")

DATA_ROOT = Path.home() / "Desktop/Projects/wmh-compression-data"
RUNS_PATH = DATA_ROOT / "runs/c1.jsonl"
TARGET_BAND = (0.60, 0.70)
TARGET_MID = 0.65
LIVE_ANCHOR_FINANCEBENCH = 0.6554  # grid llmlingua2-endpoint arm, read 2026-07-27


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT / "cache/sweep-results.json")
    data = json.loads(path.read_text())
    rows = [r for r in data["rows"] if r.get("part") in ("fit", "holdout")]
    stability = [r for r in data["rows"] if r.get("part") == "stability"]
    corpora = sorted({r["corpus"] for r in rows})
    ts = datetime.now(UTC).isoformat()
    out_lines = [
        "| corpus | calibrated t (fit) | holdout keep @t | holdout keep @0.5 | "
        "numeric ret @t vs @0.5 | identifier ret @t vs @0.5 | entity ret @t vs @0.5 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    runs_rows = []
    for corpus in corpora:
        fit = {r["threshold"]: r for r in rows if r["corpus"] == corpus and r["part"] == "fit"}
        hold = {
            r["threshold"]: r for r in rows if r["corpus"] == corpus and r["part"] == "holdout"
        }
        calibrated = min(fit, key=lambda t: abs(fit[t]["keep_subword"] - TARGET_MID))
        h_cal, h_05 = hold[calibrated], hold[0.5]
        out_lines.append(
            f"| {corpus} | {calibrated:.2f} | {h_cal['keep_subword']:.3f} | "
            f"{h_05['keep_subword']:.3f} | "
            f"{h_cal['retention']['numeric']:.3f} vs {h_05['retention']['numeric']:.3f} | "
            f"{h_cal['retention']['identifier']:.3f} vs {h_05['retention']['identifier']:.3f} | "
            f"{h_cal['retention']['entity']:.3f} vs {h_05['retention']['entity']:.3f} |"
        )
        rid = hashlib.sha1(f"{corpus}-r2-{uuid.uuid4()}".encode()).hexdigest()[:8]
        runs_rows.append(
            {
                "run_id": f"{corpus}-r2-threshold-calibration-{rid}",
                "ts": ts,
                "matrix": corpus,
                "variant": "r2-per-corpus-threshold",
                "params": {
                    "target_band": list(TARGET_BAND),
                    "scorer_fingerprint": data["fingerprint"],
                    "split_seed": 0,
                },
                "split_seed": 0,
                "fit_scenarios": fit[calibrated]["n_segments"],
                "test_scenarios": h_cal["n_segments"],
                "result": {
                    "calibrated_threshold": calibrated,
                    "holdout_keep": h_cal["keep_subword"],
                    "holdout_keep_at_global_05": h_05["keep_subword"],
                    "retention_at_calibrated": h_cal["retention"],
                    "retention_at_global_05": h_05["retention"],
                },
                "notes": "rung A offline calibration; proxies not accuracy; live leg separate",
            }
        )
        if corpus == "financebench":
            gap = abs(h_05["keep_subword"] - LIVE_ANCHOR_FINANCEBENCH)
            log.info(
                "ANCHOR CHECK financebench: holdout keep @0.5 = %.4f vs live grid %.4f "
                "(gap %.4f) -> %s",
                h_05["keep_subword"],
                LIVE_ANCHOR_FINANCEBENCH,
                gap,
                "segments represent live structure" if gap < 0.05 else "GAP TOO LARGE, investigate",
            )
    log.info("%s", "\n".join(out_lines))
    for s in stability:
        ok = s["deterministic"] and s["batch_invariant_episode"] and s["batch_invariant_segment"]
        log.info(
            "stability %s @%.2f: %s",
            s["corpus"],
            s["threshold"],
            "PASS (det + batch/append invariant)" if ok else f"FAIL {s}",
        )
    with RUNS_PATH.open("a") as f:
        for row in runs_rows:
            f.write(json.dumps(row) + "\n")
    log.info("appended %d rows to %s", len(runs_rows), RUNS_PATH)
    (DATA_ROOT / "cache/r2-calibration-table.md").write_text("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
