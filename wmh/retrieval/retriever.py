"""Retrieval over the trace replay buffer (DreamGym Eq. 4).

At each step the world model retrieves the top-k past steps whose (state, action) is most similar to
the current one, by cosine similarity of an embedding `phi`:

    {d_j} = Topk( cos( phi(s_t, a_t), phi(s_i, a_i) ) )

The buffer is initialized offline from ingested traces (`index`) and enriched online as the agent
steps (`add`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from wmh.core.render import encode_action, encode_state_action
from wmh.core.types import Action, EnvState, Observation, Step, Trace
from wmh.providers.base import Embedder

# What text phi embeds per step: the full (state, action) summary, or the command-only action.
RetrievalKey = Literal["state_action", "action"]

# A placeholder observation for query-only encoding: topk embeds (state, action), never the result.
_EMPTY_OBS = Observation(content="")


@runtime_checkable
class Retriever(Protocol):
    def index(self, traces: list[Trace]) -> None:
        """Build phase: embed every step's (state, action) and store it in the buffer."""
        ...

    def topk(self, state: EnvState, action: Action, k: int) -> list[Step]:
        """Runtime: return the k most similar prior steps to (state, action)."""
        ...

    def add(self, step: Step) -> None:
        """Online enrichment: add a freshly generated step to the buffer."""
        ...

    def sample(self, n: int) -> list[Step]:
        """Return up to `n` steps from the buffer (e.g. to seed `wmh play` action suggestions)."""
        ...


class EmbeddingRetriever:
    """Default Retriever: dense cosine similarity using a provider's embedding model.

    The replay buffer is an in-memory embedding matrix (rows = steps) kept parallel to a
    ``list[Step]``. ``index`` embeds the whole corpus in one batched ``provider.embed`` call;
    ``add`` embeds a single step for online enrichment. ``topk`` ranks by cosine similarity,
    matching DreamGym Eq. 4's ``Topk(cos(phi(s_t,a_t), phi(s_i,a_i)))``.
    """

    def __init__(self, provider: Embedder, *, key_mode: RetrievalKey = "state_action") -> None:
        self._provider = provider
        # What text phi embeds per step: "state_action" (the full (state, action) summary, default)
        # or "action" (command-only — no STATE/ACTION scaffolding, concentrating the signal for
        # stateless traces). Index and query use the SAME mode, so the buffer stays self-consistent.
        if key_mode not in ("state_action", "action"):
            raise ValueError(f"key_mode must be 'state_action' or 'action', got {key_mode!r}")
        self._key_mode = key_mode
        # Parallel structures: row i of `_matrix` is the embedding of `_steps[i]`.
        self._steps: list[Step] = []
        self._matrix: NDArray[np.float64] | None = None

    def _key_text(self, step: Step) -> str:
        if self._key_mode == "action":
            return encode_action(step.action)
        return encode_state_action(step.state_before, step.action)

    def _embed_steps(self, steps: list[Step]) -> NDArray[np.float64]:
        # phi embeds the canonical step text from wmh.core.render — the same text the engine and
        # GEPA render, so an embedded step and a shown demo match. `key_mode` selects which text.
        texts = [self._key_text(s) for s in steps]
        vectors = self._provider.embed(texts)
        return np.asarray(vectors, dtype=np.float64)

    def index(self, traces: list[Trace]) -> None:
        """Embed every step of every trace and (re)build the buffer from scratch."""
        steps = [step for trace in traces for step in trace.steps]
        self._steps = steps
        if not steps:
            self._matrix = None
            return
        self._matrix = self._embed_steps(steps)

    def topk(self, state: EnvState, action: Action, k: int) -> list[Step]:
        """Return the up-to-k most similar prior steps by cosine similarity."""
        if k <= 0 or self._matrix is None or not self._steps:
            return []
        query = self._embed_steps(
            [Step(action=action, observation=_EMPTY_OBS, state_before=state)]
        )[0]
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"embedder produces dim {query.shape[0]} but the indexed buffer has dim "
                f"{self._matrix.shape[1]}; load the same embedder (embed_dim) used at build time"
            )
        scores = _cosine(query, self._matrix)
        # argsort ascending, take the tail, reverse for descending-similarity order.
        count = min(k, len(self._steps))
        top = np.argsort(scores)[-count:][::-1]
        return [self._steps[int(i)] for i in top]

    def add(self, step: Step) -> None:
        """Append a freshly generated step to the buffer for online enrichment."""
        vector = self._embed_steps([step])
        self._steps.append(step)
        if self._matrix is None:
            self._matrix = vector
        else:
            self._matrix = np.vstack([self._matrix, vector])

    def sample(self, n: int) -> list[Step]:
        """Return the first up-to-`n` steps from the buffer (deterministic; no RNG needed)."""
        return self._steps[: max(0, n)]

    def save(self, index_dir: str | Path) -> None:
        """Persist the buffer (embedding matrix + parallel steps) under `index_dir`.

        `wmh build` writes this; `wmh serve` / `WorldModel.load` reloads it without re-embedding.
        """
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        matrix = self._matrix if self._matrix is not None else np.empty((0, 0), dtype=np.float64)
        np.save(path / _MATRIX_FILE, matrix)
        with (path / _STEPS_FILE).open("w", encoding="utf-8") as fh:
            for step in self._steps:
                fh.write(step.model_dump_json() + "\n")
        # Persist key_mode: the matrix was embedded from this mode's key text, so a reload MUST
        # query in the same mode or it cosine-compares mismatched embedding spaces (no dim error,
        # just near-random neighbours). Without this, a reloaded index reverts to state_action.
        (path / _META_FILE).write_text(json.dumps({"key_mode": self._key_mode}), encoding="utf-8")

    def load(self, index_dir: str | Path) -> None:
        """Reload a buffer previously written by `save`, replacing any current contents."""
        path = Path(index_dir)
        matrix = np.load(path / _MATRIX_FILE)
        # Split on newlines ONLY: str.splitlines() also splits on U+2028/U+2029/\x85,
        # which are legal inside JSON strings and appear verbatim in web-content
        # observations (gui-tasks corpus) — splitting there cuts a record mid-string.
        steps = [
            Step.model_validate_json(line)
            for line in (path / _STEPS_FILE).read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        self._steps = steps
        self._matrix = matrix if matrix.size and steps else None
        # Restore the mode the matrix was built with (older indexes predate meta.json -> default).
        meta_path = path / _META_FILE
        if meta_path.exists():
            mode = json.loads(meta_path.read_text(encoding="utf-8")).get("key_mode", "state_action")
            if mode not in ("state_action", "action"):
                raise ValueError(f"index meta has invalid key_mode {mode!r}")
            self._key_mode = mode


_MATRIX_FILE = "embeddings.npy"
_STEPS_FILE = "steps.jsonl"
_META_FILE = "meta.json"


def _cosine(query: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cosine similarity of `query` against each row of `matrix`. Zero vectors score 0."""
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * query_norm
    dots = matrix @ query
    # Avoid divide-by-zero: where either vector is zero, similarity is 0.
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
