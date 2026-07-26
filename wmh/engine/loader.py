"""Load a built world model from its artifact directory.

One place owns the "artifact dir -> live WorldModel" sequence (read config -> construct the serve
provider -> `WorldModel.load`). The CLI and the serving layer both call this so the loading path
stays identical no matter how the model was selected (by name, by picker, or served in bulk).
"""

from __future__ import annotations

from pathlib import Path

from wmh.config import load_config
from wmh.engine.world_model import WorldModel
from wmh.providers import provider_or_chain
from wmh.providers.base import Provider


def load_world_model(
    model_dir: str | Path,
    *,
    telemetry_root: str | Path | None = None,
    max_fidelity: bool = False,
    knowledge_dir: str | Path | None = None,
) -> tuple[WorldModel, Provider]:
    """Load the world model under `model_dir`, returning it with the serve provider it was built on.

    The provider is returned alongside so callers that also need it (e.g. `wmh demo`, which runs an
    LLM agent against the same provider) don't re-read the config or reconstruct it.
    `max_fidelity` turns on the online extras (the build-measured winner when the artifact has
    one); a plain load runs pure RAG. `knowledge_dir` overrides where the knowledge base reads and
    writes, leaving the (often read-only, shared) bundle untouched; None keeps the artifact path.
    """
    config = load_config(str(model_dir))
    provider = provider_or_chain(config.serve_provider_config())
    wm = WorldModel.load(
        str(model_dir),
        provider,
        telemetry_root=telemetry_root,
        max_fidelity=max_fidelity,
        knowledge_dir=knowledge_dir,
    )
    return wm, provider
