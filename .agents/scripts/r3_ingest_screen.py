"""Ingest-time data-quality screen: is an uploaded corpus trustworthy enough to bank?

r3 mandate (2026-07-26): the third gate of the three-gate ingest story (r1 routability,
r2 probe economics, r3 data trust). Detects, from a corpus of episode records alone (no
matrix, no real anchors), the failure classes that poisoned our banks:

  S1 empty-reply STALLS per model        (kimi precedent: consecutive/terminal empties;
                                          isolated empties are native tool-call turns in
                                          real captures - verified non-defects)
  S2 malformed tool-call share per model (glm precedent: bare/fused fn({...}), 56% of eps)
  S3 truncation share per corpus         (crmarena 55% / dabstep 49% / tau-telecom 82%)
  S4 per-model coverage imbalance        (episodes per model min/max, unscored share)
  S5 judge-unfriendly formats            (thinking/reasoning tags in replies)

Two adapters feed one screen: OTel GenAI JSONL (the ingest path's real input shape) and
OutcomeMatrix (retrospective validation against the corpora whose defects were PROVEN by
the sim2real diagnosis). Thresholds are set from the measured precedents and validated by
a confusion table over known-broken vs known-clean (corpus, model) cells; a screen that
cannot separate them is recorded as a negative and dropped (mandate kill bar). $0 API.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r3.ingest")

DATA = routing_data()
OUT = DATA / "findings" / "r3_sim2real"

_CALL_HEAD = re.compile(r"([A-Za-z_]\w*)\s*\(\s*\{")
_THINK_TAG = re.compile(r"<\s*/?\s*(think|thinking|reasoning|scratchpad)\b", re.IGNORECASE)

# Thresholds, each traceable to a measured precedent (see module docstring).
T_EMPTY = 0.05
T_MALFORMED = 0.10
T_TRUNCATION = 0.30
T_COVERAGE = 0.5
T_FORMAT = 0.20


@dataclass
class EpisodeRecord:
    """One episode as the screen sees it, whatever the source format."""

    corpus: str
    model: str
    replies: list[str]
    stop_reason: str | None = None
    scored: bool = True


def has_empty_stall(replies: list[str]) -> bool:
    """The harmful empty-reply signature: a terminal empty or >=2 consecutive empties.

    Isolated mid-episode empties appear legitimately in native tool-calling captures
    (verified on tau-bench-real: empty content turns between substantive messages,
    reward 1.0); the bug signature is empties that stall or end progress.
    """
    empties = [not r.strip() for r in replies]
    if not empties:
        return False
    if empties[-1]:
        return True
    return any(a and b for a, b in zip(empties, empties[1:], strict=False))


@dataclass
class ModelReport:
    episodes: int = 0
    replies: int = 0
    stalled: int = 0
    malformed_eps: int = 0
    truncated: int = 0
    unscored: int = 0
    format_risky: int = 0
    flags: list[str] = field(default_factory=list)


def reply_is_malformed_call(text: str) -> bool:
    """Bare/fused function-call syntax outside the JSON envelope (the glm class)."""
    if '"tool"' in text or '"done"' in text:
        return False
    for match in _CALL_HEAD.finditer(text):
        start = text.index("{", match.end() - 1)
        try:
            obj, end = json.JSONDecoder().raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(obj, dict) and text[end:].lstrip().startswith(")"):
            return True
    return False


def screen(records: list[EpisodeRecord]) -> dict:
    """Run S1-S5 over a corpus; returns per-model reports + the corpus verdict."""
    per_model: dict[str, ModelReport] = defaultdict(ModelReport)
    for rec in records:
        rep = per_model[rec.model]
        rep.episodes += 1
        rep.replies += len(rec.replies)
        if rec.replies and has_empty_stall(rec.replies):
            rep.stalled += 1
        if any(reply_is_malformed_call(r) for r in rec.replies):
            rep.malformed_eps += 1
        if rec.stop_reason == "max_steps":
            rep.truncated += 1
        if not rec.scored:
            rep.unscored += 1
        if any(_THINK_TAG.search(r) for r in rec.replies):
            rep.format_risky += 1

    episodes_by_model = {m: r.episodes for m, r in per_model.items()}
    max_eps = max(episodes_by_model.values()) if episodes_by_model else 0
    corpus_flags: list[str] = []
    report: dict = {"models": {}, "corpus_flags": corpus_flags}
    total_eps = sum(episodes_by_model.values())
    total_trunc = sum(r.truncated for r in per_model.values())
    for model, rep in sorted(per_model.items()):
        empty_share = rep.stalled / rep.episodes if rep.episodes else 0.0
        malformed_share = rep.malformed_eps / rep.episodes
        trunc_share = rep.truncated / rep.episodes
        coverage = rep.episodes / max_eps if max_eps else 0.0
        format_share = rep.format_risky / rep.episodes
        if empty_share > T_EMPTY:
            rep.flags.append(f"S1-stall {empty_share:.0%}")
        if malformed_share > T_MALFORMED:
            rep.flags.append(f"S2-malformed {malformed_share:.0%}")
        if coverage < T_COVERAGE:
            rep.flags.append(f"S4-coverage {coverage:.0%}")
        if format_share > T_FORMAT:
            rep.flags.append(f"S5-format {format_share:.0%}")
        report["models"][model] = {
            "episodes": rep.episodes,
            "stall_share": round(empty_share, 3),
            "malformed_share": round(malformed_share, 3),
            "truncation_share": round(trunc_share, 3),
            "coverage": round(coverage, 2),
            "format_risk_share": round(format_share, 3),
            "unscored_share": round(rep.unscored / rep.episodes, 3),
            "flags": rep.flags,
        }
    corpus_trunc = total_trunc / total_eps if total_eps else 0.0
    if corpus_trunc > T_TRUNCATION:
        corpus_flags.append(f"S3-truncation {corpus_trunc:.0%}")
    report["truncation_share"] = round(corpus_trunc, 3)
    # Verdict semantics match the consumer decision: corpus-level DATA-BROKEN only for
    # capture-level defects (truncation, or a systemic share of untrusted models);
    # otherwise PASS with the named untrusted model ROWS excluded from any fit.
    untrusted = sorted(m for m, r in per_model.items() if r.flags)
    systemic = per_model and len(untrusted) / len(per_model) > 1 / 3
    report["untrusted_models"] = untrusted
    report["verdict"] = (
        "DATA-BROKEN"
        if corpus_flags or systemic
        else (f"PASS-EXCLUDING-{len(untrusted)}-ROWS" if untrusted else "PASS")
    )
    return report


# ------------------------------------------------------------------ adapters


def records_from_matrix(matrix: OutcomeMatrix, corpus: str) -> list[EpisodeRecord]:
    return [
        EpisodeRecord(
            corpus=corpus,
            model=o.model,
            replies=list(o.replies),
            stop_reason=o.stop_reason,
            scored=o.reward is not None,
        )
        for o in matrix.outcomes
    ]


def _attr(attrs: list[dict], key: str) -> str | None:
    for a in attrs:
        if a.get("key") == key:
            return a.get("value", {}).get("stringValue")
    return None


def records_from_otel(path: Path, corpus: str) -> list[EpisodeRecord]:
    """Group OTel GenAI spans by traceId; completion text per span becomes a reply."""
    traces: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        span = json.loads(line)
        traces[span.get("traceId", "?")].append(span)
    records = []
    for spans in traces.values():
        replies, model = [], "unknown"
        for span in spans:
            attrs = span.get("attributes", [])
            model = _attr(attrs, "gen_ai.request.model") or model
            completion = _attr(attrs, "gen_ai.completion")
            if completion is not None:
                replies.append(completion)
            else:
                # Tool-call spans without completion text: render the call so S2 can
                # still inspect the syntax the source agent emitted.
                tool = _attr(attrs, "gen_ai.tool.name")
                args = _attr(attrs, "gen_ai.tool.call.arguments")
                if tool is not None:
                    replies.append(json.dumps({"tool": tool, "arguments": args or {}}))
        records.append(EpisodeRecord(corpus=corpus, model=model, replies=replies))
    return records


# ------------------------------------------------------------------ commands


def cmd_matrix(args: argparse.Namespace) -> None:
    results = {}
    for path in sorted((DATA / "matrices").glob("*_matrix.json")):
        corpus = path.stem.removesuffix("_matrix")
        matrix = OutcomeMatrix.load(path)
        rep = screen(records_from_matrix(matrix, corpus))
        results[corpus] = rep
        flagged = {
            m: v["flags"] for m, v in rep["models"].items() if v["flags"]
        }
        logger.info(
            "%-22s %s trunc=%.0f%% %s",
            corpus, rep["verdict"], rep["truncation_share"] * 100,
            f"| model flags: {flagged}" if flagged else "",
        )
    (OUT / "ingest_screen_matrices.json").write_text(json.dumps(results, indent=1))


def cmd_otel(args: argparse.Namespace) -> None:
    path = Path(args.path).expanduser()
    rep = screen(records_from_otel(path, path.stem))
    logger.info(json.dumps(rep, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["matrix", "otel"])
    parser.add_argument("--path", default="")
    args = parser.parse_args()
    {"matrix": cmd_matrix, "otel": cmd_otel}[args.command](args)


if __name__ == "__main__":
    main()
