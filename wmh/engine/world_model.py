"""The WorldModel: a frontier LLM acting as the environment.

This is the public API agents call (in-process or via the local backend). Each `step` retrieves
similar past steps, builds the env prompt, completes it with the serving provider, and updates the
session — including the env's free-text scratchpad "database" — to stay consistent across it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from wmh.config import ArtifactPaths, load_config
from wmh.core.parsing import dumps_observation_contract, parse_observation
from wmh.core.types import Action, EnvState, Observation, Session, Step
from wmh.engine.autoconfig import AutoFidelityReport
from wmh.engine.grounding import Grounder, extract_get_url, get_grounder, render_grounding
from wmh.engine.knowledge import KnowledgeBase
from wmh.engine.prompts import BASE_ENV_PROMPT, build_env_prompt
from wmh.optimize.gepa import VERIFY_INSTRUCTION
from wmh.optimize.reward import EpisodeRewardJudge, EpisodeScore
from wmh.providers.base import Embedder, Message, Provider
from wmh.retrieval import EmbeddingRetriever, Retriever
from wmh.retrieval.embedders import get_embedder
from wmh.telemetry import capture
from wmh.tracking import MeteredProvider, Phase, RunRecord, RunTracker

# Live web searches allowed per session (cache hits are free); keeps grounding cost bounded.
DEFAULT_GROUND_BUDGET = 5


class _StepUsage:
    """Accumulates token/cost totals across the (1 or 2) completions of a single step."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd: float | None = None


class WorldModel:
    def __init__(
        self,
        provider: Provider,
        retriever: Retriever,
        env_prompt: str = BASE_ENV_PROMPT,
        top_k: int = 5,
        telemetry_root: str | Path = ".wmh",
        reward_provider: Provider | None = None,
        *,
        knowledge: KnowledgeBase | None = None,
        reasoning: bool = False,
        grounder: Grounder | None = None,
        ground_budget: int = DEFAULT_GROUND_BUDGET,
        verify: bool = False,
        confidence: bool = False,
        confidence_why: bool = False,
        max_retrieved_observation_chars: int | None = None,
    ) -> None:
        self._provider = provider
        self._retriever = retriever
        self._env_prompt = env_prompt
        self._top_k = top_k
        self._demo_obs_cap = max_retrieved_observation_chars
        self._telemetry_root = Path(telemetry_root)
        self._sessions: dict[str, Session] = {}
        # Per-session token/cost/time accounting (serve-time observability). One tracker per
        # session, started when the session is created; `session_usage` exposes running totals.
        self._trackers: dict[str, RunTracker] = {}
        # Reward judging (`score_session`) defaults to the serve provider; pass `reward_provider`
        # to judge with a different model than the one simulating the environment.
        self._reward_provider = reward_provider or provider
        # Agentic mode (all optional; defaults preserve pre-knowledge behavior exactly).
        # `knowledge` is the cross-session KB rendered into every prompt and written to via the
        # env's kb_note; `reasoning` selects the deliberate-then-answer contract; `grounder`
        # enables bounded web search for unknown entities (None = the contract never offers it).
        self._knowledge = knowledge
        self._reasoning = reasoning
        self._grounder = grounder
        self._ground_budget = ground_budget
        self._ground_spent: dict[str, int] = {}  # session id -> live web searches used
        self._verify = verify
        # Online index enrichment (DreamGym-style): serving sessions feed generated steps back
        # into retrieval. Evaluation rollouts must NOT (see `frozen`), or one episode's
        # predictions become another's retrieved demos and results turn order-dependent.
        self._enrich_index = True
        # Verbalized confidence (WS-A6): the contract asks for a 0.0-1.0 self-assessment, carried
        # to clients in Observation.metadata. Analysis/abstention-side lever; never judged.
        self._confidence = confidence
        self._confidence_why = confidence_why

    @classmethod
    def load(
        cls,
        artifact_dir: str,
        provider: Provider,
        embedder: Embedder | None = None,
        telemetry_root: str | Path | None = None,
        reward_provider: Provider | None = None,
        max_fidelity: bool = False,
        knowledge_dir: str | Path | None = None,
    ) -> WorldModel:
        """Construct from a built `.wmh/` artifact (optimized prompt + indexed replay buffer).

        `provider` serves the live world model (generation). `embedder` supplies phi for retrieval;
        when omitted we reconstruct the configured embedder (`embed_provider` + `embed_dim`), which
        defaults to the offline `HashingEmbedder` so loading needs no embedding credentials.
        `reward_provider` backs `score_session` (defaults to `provider`) — pass it to judge with a
        different model than the one simulating the environment.

        A plain load runs pure RAG unless agentic flags were set explicitly at build
        (config.toml's `reasoning`/`knowledge`/`verify`/`grounder`). `max_fidelity=True` turns on
        the online extras: the build-measured winner from `auto_fidelity.json` when the artifact
        has one (high/max-tier builds), otherwise every extra the artifact can support.

        `knowledge_dir` overrides where the knowledge base reads and writes: when set (and
        knowledge is enabled), the KB is built from `knowledge_dir` instead of the artifact's own
        `knowledge/`, leaving the (often read-only, shared) bundle untouched. Callers seed a
        per-run temp dir and point at it; None keeps the default artifact path. It only takes
        effect when the artifact enables knowledge (config's `knowledge` flag, or `max_fidelity`):
        a plain RAG load stays pure RAG, so a seeded file in `knowledge_dir` is not rendered unless
        the model was built with knowledge on.
        """
        config = load_config(artifact_dir)
        paths = ArtifactPaths(artifact_dir)
        env_prompt = (
            paths.optimized_prompt.read_text(encoding="utf-8")
            if paths.optimized_prompt.exists()
            else BASE_ENV_PROMPT
        )
        # Reconstruct the *same* embedder the build used (provider + dim persisted in config), so
        # query vectors match the stored matrix. A caller-supplied embedder overrides.
        retriever = EmbeddingRetriever(embedder or get_embedder(config))
        if paths.index.exists():
            retriever.load(paths.index)
        reasoning, knowledge_on, verify, grounder_kind = (
            config.reasoning,
            config.knowledge,
            config.verify,
            config.grounder,
        )
        top_k_override: int | None = None
        demo_obs_cap: int | None = None
        if max_fidelity:
            if paths.auto_fidelity.exists():
                winner = AutoFidelityReport.model_validate_json(
                    paths.auto_fidelity.read_text(encoding="utf-8")
                ).winner
                reasoning, knowledge_on, verify, grounder_kind = (
                    winner.reasoning,
                    winner.knowledge,
                    winner.verify,
                    winner.grounder,
                )
                top_k_override, demo_obs_cap = winner.top_k, winner.demo_obs_cap
            else:
                # No measured winner (low/medium builds): all the extras the artifact supports.
                reasoning, verify = True, True
                knowledge_on = _knowledge_dir(paths, knowledge_dir).is_dir()
                grounder_kind = "fetch" if grounder_kind == "none" else grounder_kind
        effective_knowledge_dir = _knowledge_dir(paths, knowledge_dir)
        return cls(
            provider,
            retriever,
            env_prompt=env_prompt,
            top_k=top_k_override if top_k_override is not None else config.top_k,
            telemetry_root=telemetry_root or _default_telemetry_root(artifact_dir),
            reward_provider=reward_provider,
            knowledge=(
                KnowledgeBase(effective_knowledge_dir)
                if knowledge_on and effective_knowledge_dir.is_dir()
                else None
            ),
            reasoning=reasoning,
            grounder=None if grounder_kind == "none" else get_grounder(grounder_kind),
            verify=verify,
            confidence=config.confidence,
            confidence_why=config.confidence_why,
            max_retrieved_observation_chars=demo_obs_cap,
        )

    def new_session(
        self,
        task: str | None = None,
        seed_state: EnvState | None = None,
        *,
        enrich: bool = True,
    ) -> Session:
        """Open a session. `enrich=False` keeps its steps out of the shared retrieval buffer —
        required for evaluation rollouts, whose PREDICTED observations must not become demos for
        later rollouts (order-dependent, self-reinforcing scores otherwise)."""
        session = Session(
            id=uuid.uuid4().hex, task=task, state=seed_state or EnvState(), enrich=enrich
        )
        self._sessions[session.id] = session
        tracker = RunTracker(run_id=session.id, kind="serve")
        tracker.start()
        self._trackers[session.id] = tracker
        capture(
            "wmh generated trace started",
            {"generated_trace_count": 1},
            root=self._telemetry_root,
        )
        return session

    def session_usage(self, session_id: str) -> RunRecord:
        """Return the running token/cost/time record for `session_id` (DreamGym serve metering).

        The tracker is started at `new_session` and intentionally left running for the life of the
        session (serve sessions have no explicit end), so `duration_seconds` is live wall-clock
        since the session was created — not just time spent inside `step`. Raises `KeyError` if
        `session_id` is unknown (callers that take session ids from clients should validate first,
        as the serving API does).
        """
        return self._trackers[session_id].record_summary()

    def get_session(self, session_id: str) -> Session:
        return self._sessions[session_id]

    @property
    def knowledge(self) -> KnowledgeBase | None:
        """The cross-session knowledge base, or None when the artifact ships none."""
        return self._knowledge

    def end_session(self, session_id: str) -> RunRecord:
        """End `session_id`: stop its metering, drop it from the model, return the final usage.

        Batch rollouts open thousands of sessions against one WorldModel; without an explicit end,
        every Session (full step history) and its still-ticking RunTracker stays resident forever.
        Raises `KeyError` for an unknown/already-ended session id.
        """
        self._sessions.pop(session_id)  # KeyError for unknown ids, same as get_session
        self._ground_spent.pop(session_id, None)
        tracker = self._trackers.pop(session_id)
        tracker.stop()
        return tracker.record_summary()

    def sample_steps(self, n: int) -> list[Step]:
        """Return up to `n` steps from the replay buffer (used to seed `wmh play` suggestions)."""
        return self._retriever.sample(n)

    def score_session(self, session_id: str) -> EpisodeScore:
        """Judge the session's rollout so far: episode reward, per-step rewards, and a critique.

        This is the RL reward signal. The reward judge sees the session's task and its full step
        history — never a gold trace — and returns everything the algorithms need in one call:
        scalar reward/success (GRPO, PPO, REINFORCE++), `step_rewards` (dense diagnostics), and
        `critique` (SDPO's teacher feedback). Raises `KeyError` for an unknown session id.

        Judge usage is metered onto the session's tracker under `Phase.JUDGE`, so `session_usage`
        keeps reward cost separate from the world model's SERVE cost.
        """
        session = self._sessions[session_id]
        tracker = self._trackers.get(session_id)
        provider = (
            MeteredProvider(self._reward_provider, tracker, base_phase=Phase.JUDGE)
            if tracker is not None
            else self._reward_provider
        )
        score = EpisodeRewardJudge(provider).score(session.task, session.history)
        # The scalar rides the final observation too, so replay-buffer consumers that read
        # Observation.reward (DreamGym-style terminal r) see the same number the API returned.
        if session.history:
            session.history[-1].observation.reward = score.reward
        return score

    @contextmanager
    def frozen(self) -> Iterator[WorldModel]:
        """Suspend online index enrichment for the duration of the block.

        Evaluation rollouts (scenario verification, score matrices) step the world model many
        times; if those generated steps were indexed, later episodes would retrieve earlier
        episodes' predictions instead of only the built trace corpus — results would depend on
        evaluation order. Serving resumes enrichment when the block exits.
        """
        previous = self._enrich_index
        self._enrich_index = False
        try:
            yield self
        finally:
            self._enrich_index = previous

    def render_step_prompt(self, session_id: str, action: Action) -> str:
        """Assemble the exact (system + user) env prompt `step` would send, without calling the LLM.

        Used by `wmh demo` to display what the world model sees. Read-only: no session mutation.
        """
        session = self._sessions[session_id]
        demos = self._retriever.topk(session.state, action, self._top_k)
        system, user = build_env_prompt(
            self._env_prompt,
            session,
            action,
            demos,
            knowledge=self._rendered_knowledge(),
            reasoning=self._reasoning,
            grounding=self._grounder is not None,
            confidence=self._confidence,
            confidence_why=self._confidence_why,
            max_retrieved_observation_chars=self._demo_obs_cap,
        )
        return f"{system}\n\n=== USER ===\n{user}"

    def step(self, session_id: str, action: Action) -> Observation:
        """Predict the observation for `action` and advance the session. DreamGym Eq. (4)."""
        started = time.monotonic()
        session = self._sessions[session_id]

        # (1) retrieve top-k similar past steps conditioned on the latest state + action
        demos = self._retriever.topk(session.state, action, self._top_k)

        # (2) assemble the env prompt and (3) predict the observation — with at most one bounded
        # grounding round-trip (the model asks to search the web for an entity it can't ground,
        # then answers again with the results in context)
        usage = _StepUsage()
        try:
            observation = self._predict(session, action, demos, usage)
        except Exception:
            capture(
                "wmh generated step failed",
                {
                    "success": False,
                    "duration_seconds": round(time.monotonic() - started, 3),
                },
                root=self._telemetry_root,
            )
            raise

        # Metering happened per completion inside _complete; advance appends the step,
        # folds the scratchpad note + cross-session kb_note, and enriches the buffer.
        self._advance(session, action, observation)
        capture(
            "wmh generated step completed",
            {
                "success": True,
                "generated_step_count": 1,
                "session_step_count": len(session.history),
                "duration_seconds": round(time.monotonic() - started, 3),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": round(usage.cost_usd, 6) if usage.cost_usd is not None else None,
            },
            root=self._telemetry_root,
        )
        return observation

    def step_open_loop(self, session_id: str, action: Action, actual: Observation) -> Observation:
        """Predict like `step`, but advance the session with the RECORDED observation.

        Teacher-forced replay: the prediction is returned for display/scoring while the session
        continues from ground truth, so later predictions are conditioned on the real trajectory
        (the open-loop protocol used by `wmh demo` and the replay eval).
        """
        session = self._sessions[session_id]
        demos = self._retriever.topk(session.state, action, self._top_k)
        usage = _StepUsage()
        prediction = self._predict(session, action, demos, usage)
        # Advance from GROUND TRUTH only: the prediction must never reach the session state,
        # the cross-session KB, or the retrieval buffer — a hallucinated kb_note written to
        # learned.md here would render into every future prompt as canonical fact.
        self._advance(session, action, actual)
        return prediction

    def seed_session(self, session_id: str, steps: list[Step]) -> None:
        """Advance a session with already-recorded steps, no prediction (open-loop resume).

        Used when a replay continues on a fresh WorldModel (e.g. after a provider switch):
        teacher-forced history is the recorded trajectory, so seeding is just advancing.
        """
        session = self._sessions[session_id]
        for step in steps:
            self._advance(session, step.action, step.observation)

    def _advance(self, session: Session, action: Action, observation: Observation) -> None:
        """Append the step, fold its state note into the scratchpad, and enrich the buffer.

        `state_before` is a deep copy: `_update_state` mutates `session.state` in place, and an
        aliased reference would rewrite every recorded step (and the text the retriever embeds)
        to the post-mutation state.
        """
        step = Step(
            action=action,
            observation=observation,
            state_before=session.state.model_copy(deep=True),
            task=session.task,
        )
        session.history.append(step)
        self._update_state(session, step)
        # Enrich the shared retrieval buffer + cross-session KB only when the model enriches
        # online AND this session opts in — eval/closed-loop rollouts pass enrich=False so
        # predicted steps and kb_notes never leak into retrieval or the knowledge base.
        if self._enrich_index and session.enrich:
            self._update_knowledge(session, observation)
            self._retriever.add(step)

    def _predict(
        self, session: Session, action: Action, demos: list[Step], usage: _StepUsage
    ) -> Observation:
        """One observation prediction, with at most one grounding search + re-completion.

        Two grounding paths, both budgeted through `_ground`:
        - PREFETCH: when the action is itself a read-only `curl` GET, the URL is fetched before
          the first completion (no extra provider call) — found empirically to be the dominant
          groundable case (42% of terminal-tasks steps).
        - ground_query: the model may still request a search for entities the prefetch can't see;
          the re-completion renders with `grounding=False` so the step is bounded at two calls.
        """
        knowledge = self._rendered_knowledge()
        if self._grounder is not None:
            url = extract_get_url(action)
            if url is not None:
                fetched = self._ground(session.id, url)
                if fetched is not None:
                    block = f"## live fetch: {url}\n{fetched}"
                    knowledge = f"{knowledge}\n\n{block}" if knowledge else block
        observation = self._complete(
            session, action, demos, knowledge, self._grounder is not None, usage
        )
        query = observation.metadata.get("ground_query")
        if self._grounder is not None and isinstance(query, str) and query.strip():
            results = self._ground(session.id, query.strip())
            if results is not None:
                grounded_section = f"## web search results: {query.strip()}\n{results}"
                knowledge = f"{knowledge}\n\n{grounded_section}" if knowledge else grounded_section
                observation = self._complete(session, action, demos, knowledge, False, usage)
        if self._verify:
            observation = self._verify_draft(session, action, demos, knowledge, observation, usage)
        return observation

    def _verify_draft(
        self,
        session: Session,
        action: Action,
        demos: list[Step],
        knowledge: str | None,
        draft: Observation,
        usage: _StepUsage,
    ) -> Observation:
        """Second self-check completion: the draft is re-examined against the full evidence.

        The `verify` mode (~2x serve cost): re-present the exact prompt plus the draft and ask
        for a correction (or the unchanged draft). Empirically the best configuration where
        content prediction is hardest (swe-style suites) and noise elsewhere — enable per task,
        or let `wmh build --max-fidelity` decide.
        """
        system, user = build_env_prompt(
            self._env_prompt,
            session,
            action,
            demos,
            knowledge=knowledge,
            reasoning=self._reasoning,
            grounding=False,
            confidence=self._confidence,
            confidence_why=self._confidence_why,
            max_retrieved_observation_chars=self._demo_obs_cap,
        )
        verify_user = user + VERIFY_INSTRUCTION.format(draft=dumps_observation_contract(draft))
        completion = self._provider.complete(system, [_user_message(verify_user)])
        usage.input_tokens += completion.usage.input_tokens
        usage.output_tokens += completion.usage.output_tokens
        tracker = self._trackers.get(session.id)
        if tracker is not None:
            # Prefer the model that actually served (failover chains set completion.model).
            served = completion.model or self._provider.config.model
            event = tracker.record(Phase.SERVE, served, completion.usage)
            if event.cost_usd is not None:
                usage.cost_usd = (usage.cost_usd or 0.0) + event.cost_usd
        return parse_observation(completion.text)

    def _complete(
        self,
        session: Session,
        action: Action,
        demos: list[Step],
        knowledge: str | None,
        grounding: bool,
        usage: _StepUsage,
    ) -> Observation:
        """One metered provider completion under the current agentic-mode settings."""
        system, user = build_env_prompt(
            self._env_prompt,
            session,
            action,
            demos,
            knowledge=knowledge,
            reasoning=self._reasoning,
            grounding=grounding,
            confidence=self._confidence,
            confidence_why=self._confidence_why,
            max_retrieved_observation_chars=self._demo_obs_cap,
        )
        completion = self._provider.complete(system, [_user_message(user)])
        usage.input_tokens += completion.usage.input_tokens
        usage.output_tokens += completion.usage.output_tokens
        tracker = self._trackers.get(session.id)
        if tracker is not None:
            # Prefer the model that actually served (failover chains set completion.model).
            served = completion.model or self._provider.config.model
            event = tracker.record(Phase.SERVE, served, completion.usage)
            if event.cost_usd is not None:
                usage.cost_usd = (usage.cost_usd or 0.0) + event.cost_usd
        return parse_observation(completion.text)

    def _rendered_knowledge(self) -> str | None:
        """The KB rendered for the prompt, or None when the KB is absent/empty."""
        if self._knowledge is None:
            return None
        return self._knowledge.render() or None

    def _ground(self, session_id: str, query: str) -> str | None:
        """Resolve a ground_query: KB cache first, then one budgeted live search.

        Cache hits are free (no budget spend). A live search is cached into the KB
        (`grounded.md`) so the same entity is never searched twice across sessions. Returns None
        when the session's live-search budget is exhausted (the step proceeds ungrounded).
        """
        if self._knowledge is not None:
            cached = self._knowledge.lookup_grounded(query)
            if cached is not None:
                return cached
        if self._grounder is None or self._ground_spent.get(session_id, 0) >= self._ground_budget:
            return None
        try:
            results = self._grounder.ground(query)
        except Exception:  # noqa: BLE001 - a search-API hiccup degrades to ungrounded, never 500s
            return None
        self._ground_spent[session_id] = self._ground_spent.get(session_id, 0) + 1
        results_text = render_grounding(results)
        # Cache HITS only: persisting "(no results)" would negative-cache a transient failure
        # into the artifact forever (lookup_grounded returns it as a truthy hit).
        if results and self._knowledge is not None:
            self._knowledge.append_grounded(query, results_text)
        return results_text

    def _update_knowledge(self, session: Session, observation: Observation) -> None:
        """Persist the env's `kb_note` (a cross-session canonical fact) to the KB.

        Writes go only to `learned.md` with session provenance — seeded files and human edits are
        never auto-modified.
        """
        if self._knowledge is None:
            return
        note = observation.metadata.get("kb_note")
        if isinstance(note, str) and note.strip():
            self._knowledge.append_learned(note, provenance=f"session {session.id[:8]}")

    def _update_state(self, session: Session, step: Step) -> None:
        """Fold the step's effect into session.state (the env's free-text scratchpad "database").

        The world model emits a one-line `state_note` (carried in `observation.metadata`) describing
        what changed; we append it to the scratchpad so later steps in the session stay consistent
        (e.g. "booked r_900 for u_kath"). Structured state is left to explicit seeding/tooling.
        """
        note = step.observation.metadata.get("state_note")
        if isinstance(note, str) and note.strip():
            prefix = f"{session.state.scratchpad}\n" if session.state.scratchpad else ""
            session.state.scratchpad = f"{prefix}- {note.strip()}"
        # The environment PROFILE is a revised belief state, not a log: `state_update` REPLACES
        # it wholesale, so beliefs contradicted by this step (a killed server, an overwritten
        # file) disappear instead of surviving as stale scratchpad lines.
        update = step.observation.metadata.get("state_update")
        if isinstance(update, str) and update.strip():
            session.state.structured["profile"] = update.strip()


def _user_message(text: str) -> Message:
    return Message(role="user", content=text)


def _knowledge_dir(paths: ArtifactPaths, override: str | Path | None) -> Path:
    """The effective knowledge directory: the caller's override when given, else the artifact's."""
    return Path(override) if override is not None else paths.knowledge


def _default_telemetry_root(artifact_dir: str | Path) -> Path:
    path = Path(artifact_dir)
    if path.parent.name == "models":
        return path.parent.parent
    return path
