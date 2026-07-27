"""Locate the routing research corpus (`runs/`, `matrices/`, `cache/`, `findings/`).

The corpus is multi-GB measured data that git does not carry, so every research script has to
find it at runtime rather than read it from the tree. Before the wmo rename each script hardcoded
one contributor's absolute path; that made the scripts unrunnable for anyone else and pinned a
repo name that no longer exists. `.agents/scripts/validate_knn_promotion.py` established the
convention this module centralises: default to the gitignored `.wmo/routing-data/` artifact root
of whatever checkout is running, and let `$WMO_ROUTING_DATA` point somewhere else.

Two entry points because the callers differ. Scripts want `routing_data()`, which fails loudly
and immediately when the corpus is absent — a research run that silently reads no rows is worse
than one that refuses to start. Tests want `routing_data_root()`, which only resolves the path,
so a corpus-backed test can `skipif` on it instead of erroring at collection time on a machine
that has no corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_ROUTING_DATA = "WMO_ROUTING_DATA"
# Mirrors validate_knn_promotion.py: the corpus is untracked research data, so it defaults to the
# gitignored .wmo/ artifact root of this checkout. parents[2] is the repo root from wmo/research/.
DEFAULT_ROUTING_DATA = Path(__file__).resolve().parents[2] / ".wmo" / "routing-data"


def routing_data_root() -> Path:
    """Where the corpus is expected, whether or not anything is actually there."""
    override = os.environ.get(ENV_ROUTING_DATA)
    return Path(override).expanduser() if override else DEFAULT_ROUTING_DATA


def routing_data() -> Path:
    """Root of the routing research corpus, holding `runs/`, `matrices/`, and `cache/`.

    Raises:
        SystemExit: if the corpus is not where the environment says it is, naming the
            directory that is missing and the variable that redirects the lookup.
    """
    root = routing_data_root()
    if not root.is_dir():
        raise SystemExit(
            f"routing corpus not found at {root}. It is multi-GB research data that git does "
            f"not carry: set {ENV_ROUTING_DATA} to the directory holding runs/, matrices/, and "
            "cache/, or place the corpus at that default path."
        )
    return root
