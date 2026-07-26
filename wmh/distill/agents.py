"""Harbor agent bridge for distillation rollouts with token-span capture.

Harbor instantiates `WmhDistillHarborAgent` (via
`WMH_DISTILL_HARBOR_AGENT_IMPORT_PATH` plus JSON kwargs) for every distillation
trial. The one behavioral difference from the base `WmhHarborAgent` is
provider construction: a Tinker worker provider is built with a `TokenRecorder`
that writes the trial's exact sampled token spans to
`{token_sink_dir}/{trial_name}.jsonl`.

A batch that only needs GENERATION takes the base bridge's plain provider
instead (no recorder, `token_sink_path` stays None): a served
OpenAI-compatible teacher (`teacher.backend = "openai_compat"`) cannot report
the token ids it sampled, but the teacher-in-harness episodes it runs (the
gate's holdout baseline) are scored on verifier rewards alone and never
trained on. The paths that DO train on spans refuse a non-tinker provider up
front, before harbor spends anything (`wmh.distill.loop`).

The sink lives OUTSIDE harbor's trial directory on purpose: the scorer's
entry prune deletes invalid trial dirs wholesale before re-running them, and
a sink inside the trial dir would vanish with it. The rollout collector keys
the sink dir per training step and joins sinks back to trials by name.

Like every module in `wmh.evals.harbor`, this module imports the harbor SDK
at module scope and is therefore imported lazily by its consumers; `import
wmh.distill` must succeed without the harbor extra.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from harbor.models.task.config import MCPServerConfig

from wmh.core.types import JsonObject
from wmh.evals.harbor.agent import (
    DEFAULT_EPISODE_WORKERS,
    MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
    WmhHarborAgent,
)
from wmh.harness.runtime import DEFAULT_EVAL_EPISODE_TIMEOUT_S
from wmh.providers.base import Provider, ProviderConfig, ProviderKind
from wmh.providers.retry import wrap_provider_with_retries
from wmh.providers.tinker import TinkerChatProvider, TokenRecorder

WMH_DISTILL_HARBOR_AGENT_IMPORT_PATH = "wmh.distill.agents:WmhDistillHarborAgent"

logger = logging.getLogger(__name__)


class WmhDistillHarborAgent(WmhHarborAgent):
    """Runs the candidate with a Tinker provider that records its token spans.

    Everything else (episode execution, trace persistence, cancellation
    semantics) is inherited from `WmhHarborAgent` unchanged.

    Args:
        token_sink_dir: Directory the per-trial span sink is written into; the
            sink file is named `{trial_name}.jsonl` where the trial name is
            derived from harbor's logs-dir layout (`{trial_dir}/agent`).
    """

    token_sink_path: Path | None
    """The exact sink file this trial's recorder writes (set at construction),
    or None for a generation-only trial whose provider records no spans."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        *,
        token_sink_dir: str,
        command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        extra_env: dict[str, str] | None = None,
        harness: JsonObject,
        provider_config: JsonObject,
        harness_backend: Literal["local", "e2b"] = "local",
        e2b_template: str | None = None,
        episode_timeout_sec: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        episode_workers: int = DEFAULT_EPISODE_WORKERS,
        context_window: int | None = None,
    ) -> None:
        if not isinstance(token_sink_dir, str) or not token_sink_dir:
            raise ValueError(
                "token_sink_dir must be a nonempty path string; the distill rollout "
                "collector passes it through the scorer's extra_agent_kwargs"
            )
        # Set before super().__init__: the base constructor calls _build_provider,
        # which reads this (and the BaseAgent-assigned logs_dir).
        self._token_sink_dir = Path(token_sink_dir)
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            command_timeout_sec=command_timeout_sec,
            extra_env=extra_env,
            harness=harness,
            provider_config=provider_config,
            harness_backend=harness_backend,
            e2b_template=e2b_template,
            episode_timeout_sec=episode_timeout_sec,
            episode_workers=episode_workers,
            context_window=context_window,
        )

    def _build_provider(self, config: ProviderConfig) -> Provider:
        """Build this trial's worker provider, recording spans when it can.

        Two explicit paths, keyed by the provider kind, because only the Tinker
        provider reports the exact token ids it sampled:

        - `tinker` (the student, or a Tinker teacher): wrapped with a
          `TokenRecorder` writing this trial's spans to
          `{token_sink_dir}/{trial_name}.jsonl`, which is what training reads.
        - any other kind: a generation-only trial (a served OpenAI-compatible
          teacher producing the gate's holdout baseline). Built exactly like the
          base bridge, with no recorder and no sink file. Callers that train on
          spans refuse these providers before the batch runs, so reaching this
          branch means the batch is scored on verifier rewards alone.

        Args:
            config: The validated worker provider config.

        Returns:
            The retry-wrapped provider. Spans record only after a completion
            fully succeeds, so retries never duplicate them.

        Raises:
            ValueError: If the trial name cannot be derived from the logs dir.
        """
        if config.kind is not ProviderKind.TINKER:
            logger.info(
                "trial %s samples %s (kind %s) without token capture: only the Tinker "
                "provider can record sampled token ids, so this trial contributes "
                "verifier reward only",
                self.logs_dir.parent.name,
                config.model,
                config.kind.value,
            )
            self.token_sink_path = None
            return super()._build_provider(config)
        trial_name = self.logs_dir.parent.name
        if not trial_name:
            raise ValueError(
                f"cannot derive the harbor trial name from logs_dir {self.logs_dir}; "
                "the distill agent expects harbor's per-trial {trial_dir}/agent layout"
            )
        self._token_sink_dir.mkdir(parents=True, exist_ok=True)
        sink_path = self._token_sink_dir / f"{trial_name}.jsonl"
        # The recorder appends and call_index restarts at 0 per recorder; a leftover
        # sink from a pruned earlier attempt of the same trial name would corrupt the
        # contiguous call_index sequence load_trial_spans enforces.
        sink_path.unlink(missing_ok=True)
        self.token_sink_path = sink_path
        provider = TinkerChatProvider(config, recorder=TokenRecorder(jsonl_path=sink_path))
        # Same retry contract as the base bridge: one transient capacity error must
        # not kill a whole trial.
        return wrap_provider_with_retries(provider)
