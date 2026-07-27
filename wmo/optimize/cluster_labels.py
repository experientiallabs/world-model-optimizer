"""Human-readable cluster labels for the routing request log.

A fitted policy's `cluster_label` is a product surface: the request log shows WHY a request
went where ("rank router: nearest cluster 12 (airline-cancellation refund)"). Two sources,
best first:

1. Majority scenario-id prefix ("mmlu-anatomy", "tau-bench") when ids carry `prefix:` task
   names; it is exact provenance.
2. c-TF-IDF terms otherwise (BERTopic's class-based TF-IDF, arXiv 2203.05794, simplified):
   score each term by its in-cluster frequency times log(1 + K / cluster-df), take the top
   `max_terms`. Distinctive words beat frequent words, so two retail clusters get different
   labels ("exchange camera zoom" vs "cancel order refund") instead of both saying "order".

Pure text processing, no model calls; labeling never affects selection.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z][a-z0-9_-]{2,}")
# Function words plus the JSON-scaffolding vocabulary of wm task payloads; anything this
# generic would label every cluster identically.
_STOP = frozenset(
    """the and for you your are with that this not have has can will its all any out from
    into over under been being what when where which while about after before because but
    they them their there then than each such only also more most some very just like want
    need make sure ask say tell get use used using known info reason call task domain
    instructions user known_info reason_for_call should would could must may might his her
    him she who whom does did doing done was were is it in on at to of as by an or if do be
    no so we he up""".split()
)


def tokenize(text: str) -> list[str]:
    # Literal escape sequences inside JSON-encoded task payloads ("...\nYou are...") would
    # otherwise fuse into pseudo-words like "nyou".
    text = text.replace("\\n", " ").replace("\\t", " ").lower()
    return [t for t in _TOKEN.findall(text) if t not in _STOP]


def label_clusters(cluster_texts: list[list[str]], *, max_terms: int = 3) -> list[str]:
    """One c-TF-IDF label per cluster (see module docstring). Empty list of texts -> ""."""
    k = len(cluster_texts)
    term_counts = [Counter(t for text in texts for t in tokenize(text)) for texts in cluster_texts]
    df = Counter(term for counts in term_counts for term in counts)
    labels = []
    for counts in term_counts:
        total = sum(counts.values())
        if not total:
            labels.append("")
            continue
        scored = {
            term: (freq / total) * math.log(1 + k / df[term]) for term, freq in counts.items()
        }
        top = sorted(scored, key=lambda t: (-scored[t], t))[:max_terms]
        labels.append(" ".join(top))
    return labels


def majority_prefix(scenario_ids: list[str]) -> str:
    """The most common `prefix:` of the ids, or "" when ids carry no prefix."""
    prefixes = Counter(sid.split(":", 1)[0] for sid in scenario_ids if ":" in sid)
    return prefixes.most_common(1)[0][0] if prefixes else ""
