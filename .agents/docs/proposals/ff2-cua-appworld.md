# FF2 — CUA world model vs a coded ground-truth oracle (AppWorld v1 + visual leg)

*Proposal, 2026-07-05. Supersedes the greenfield-first draft (parked on `ff-cua/deskbook-v0`).
User decisions (2026-07-05): (1) start with the research-verified external env — AppWorld — and
add a second, more visual benchmark leg (adopt one if adoptable, else build our own);
(2) text observations on AppWorld v1, images as the stated future direction; (3) gate-accuracy
headline with open- and closed-loop measurement; (4) launch = docs page + side-by-side GIF.
Mirrored to `~/Desktop/Projects/wmh-plan/plans/ff2.md`; cross-chat log: DECISIONS.md.*

## Thesis (the launch argument)

Computer-use agents need private, resettable environments; the fashionable answer is a
"neural OS" — a model hallucinating screens — with no way to know when it's wrong. Three 2026
papers frame the moment:

- **CUWM** (Microsoft, arXiv 2602.17365): desktop world model for Office; factorizes UI dynamics
  into a *textual transition description* then a *visual realization* (next-screenshot
  synthesis). Evaluated on real apps — which expose **no programmatic ground-truth state** — so
  correctness is judged by similarity and downstream decision quality.
- **gWorld** (arXiv 2602.01576): mobile GUI world model that generates the next UI as
  **executable web code** instead of pixels — code as the render substrate for predicted state.
- **Dockerless** (arXiv 2606.28436): environment-free verifier for coding agents, *validated
  against execution-based ground truth* before replacing it.

We invert the neural-OS direction the same way Dockerless does for verification: keep a **real
environment built in code** (real state, real rules) as the oracle, build the world model from
traces of it, and score the world model's correctness **exactly** against the oracle — per
API outcome, per state transition, per gate decision. Neural-OS approaches cannot run this
benchmark; that asymmetry is the launch.

CUWM's own factorization concedes the point that matters: *what changes* (state, textual) is
predicted first and *how it appears* (pixels) is rendered second. We benchmark the first layer —
where correctness is definable — and treat pixels as presentation (future direction, stated).

## Leg 1 (v1): AppWorld as the oracle

Verified properties (research pass, 2026-07-02): Python simulator, 9 apps / 457 APIs / 750 tasks
over per-app SQLite; `world.save_state()/load_state()`; sub-second warm resets; typed model
accessors (`models.<app>.<Model>.find_*`, `models.changed_model_names()`); state-based
`world.evaluate()`; Apache-2.0; `pip install appworld`; ACL 2024 best resource paper. Caveats:
pin a known-good version (recent regressions: issues #199/#203/#204/#206); test-split ground
truth ships encrypted → **we use train/dev tasks only**; verify determinism empirically (no
wall-clock/RNG leak) before trusting replay.

### Environment adapter

`examples/appworld/` wraps AppWorld behind the `wmh.env.Env` protocol (D9/D22 seam):

- **Action** = one API call: `tool_call` with `name="<app>.<api>"`,
  `arguments=<call args>` (AppWorld's function-calling surface, not its code-REPL mode — one
  call per step keeps steps aligned with the harness's step unit and with gate scoring).
- **Observation** = the oracle's real JSON response (`content`), `is_error` from the API
  error flag.
- **`wmh.state.structured`** = a compact per-step snapshot: current app/session context +
  `changed_model_names()` since the previous step + record counts for the task's relevant
  apps. Never the full DB (too big, and leak-prone).
- Eval-side ground truth (NOT world-model-visible): per-step gate labels derived from the
  response (see taxonomy) + the concrete DB diff for mutating calls. Rides on
  `wmh.trace.metadata` / a `cua.gates` span attribute the ingest adapter ignores — enforced by
  a leak test asserting no gate field reaches ingested Steps or the RAG index.

### Gate taxonomy (the coded scorer's classes)

State-dependent API outcomes, labeled deterministically from the oracle response + state diff:

| family | examples in AppWorld |
|---|---|
| auth | unauthenticated call, wrong credentials, expired/invalid token |
| validation | malformed/missing arguments, format rules |
| not_found | id references a record that doesn't exist (state-dependent!) |
| permission / business_rule | acting on another user's resource; insufficient balance; duplicate; app-specific rules |
| state_effect | for mutating calls: which models changed (created/updated/deleted) vs oracle's diff |
| success | well-formed call on valid state succeeds with the right shape |

**GateScorer** (`gate_scorer.py`, deterministic, no LLM): extract the gate outcome from the
world model's *predicted* observation (error flag + error class parsed from the predicted JSON),
compare to the oracle label. Gate accuracy = agreement rate, reported per family. A
self-consistency test asserts the extractor reproduces the oracle's own labels on every recorded
step before it's ever pointed at predictions (AGENTS rule 12: read real scored outputs before
trusting the number).

### Metrics

1. **Open-loop gate accuracy (headline)** — teacher-forced replay per D12; GateScorer over
   predicted vs actual outcomes, per gate family. The Z5 hypothesis says this is where WMs
   crack; the per-family table is the story either way.
2. **Closed-loop task-success agreement** — same agent, identical reset, run once against the
   oracle and once against the WM (`run_episode`, byte-identical loop via the Env seam);
   AppWorld's `evaluate()` scores the oracle run; report (a) does the WM-run episode *claim*
   the same success story, (b) steps-until-first-gate-mismatch (shares FF3's divergence-horizon
   framing; reuse `wmh/research/divergence.py` if landed).
3. **Open-loop rubric fidelity** (`evals/default.toml`, D12 conventions) for continuity with
   tau/terminal/swe numbers. NOTE: judge conventions changed while FF2 was paused (PR #83 judge
   overhaul, fidelity scales not comparable) — re-check DECISIONS.md/mainline judge state at
   implementation and report on whatever the current mainline scale is.

### Corpus

Train/dev tasks only (test GT encrypted). Target ≈ 100–150 tasks, 1 trace each + natural retry
noise, ~1,500–3,000 steps — trace-scaling says fidelity saturates in tens of traces; what matters
is **coverage of the gate taxonomy**, so capture is stratified: task selection + a few
deliberately-adversarial instructions per family (wrong ids, unauthenticated attempts) to make
denied-gates common, not rare. Capture agent: Bedrock (Opus sharded 4.6/4.7/4.8, us-east-1),
function-calling loop through the Env adapter; heavy deps in the example venv (`run.sh`).
Converter: stdlib-only → `traces.otel.jsonl` (terminal-tasks pattern + `wmh.state.structured`).

### Phases

- **P0** env adapter + gate labeler + tests (incl. determinism check + leak test) — failing
  tests first.
- **P1** capture (smoke 10 tasks → read them → full run) + converter + committed corpus.
- **P2** `wmh build` (base + RAG), fidelity suite, GateScorer over replay; read scored samples.
- **P3** closed-loop: oracle-vs-WM paired episodes, success agreement + divergence.
- **P4** launch page `docs/research/cua_world_model_oracle.md` + side-by-side GIF (same episode
  stepping oracle vs WM, red flash at first gate mismatch) + honest cost table.
- **P5** `/ready-for-merge`, DECISIONS.md updates (gate metric offered to the benchmark chats).

## Leg 2: the visual benchmark (decided: yes; WHICH is the open question)

User: "add another modality through another more visual benchmark (and if we don't have then
we'll need to yes create our own)". Candidates, researched:

- **(a) Build our own, gWorld-style (recommended):** revive the parked deskbook oracle
  (`ff-cua/deskbook-v0`: gate-heavy mini-CRM/booking app, coded state machine, a11y renderer,
  Env backend, 28 tests — built during the AFK window, ~1 day from visual). Add a
  deterministic **HTML renderer of the same state** — the UI *is* code (gWorld's substrate), so
  the oracle stays exactly inspectable AND every state renders to a real screen for the GIF and
  future image-observation work. Full gate control; zero external deps.
- **(b) MiniWoB++** (MIT, `pip install miniwob`, seeded Gymnasium resets, DOM + pixels):
  genuinely visual and light, but tasks are toy single-screens; weak gate story.
- **(c) VisualWebArena / WebArena-Lite** (real web apps, real DOM+pixels): credibility, but
  ~1 TB Docker stack, minutes-long resets, cross-task contamination — fails the cheap-reset bar
  for a benchmark meant to be run by others.
- **(d) gWorld's own mobile-GUI eval sets**: offline transition datasets, not interactive
  environments — no gates, no closed loop; usable only as an open-loop extra.

## Coordination (cross-chat)

- **AppWorld claim**: WS-B2's D22 put appworld in its Wave 3 ("after real upstream data") via
  SIB synthetic seeds. FF2 now builds `examples/appworld/` from the REAL upstream env on user
  direction — logged in DECISIONS.md; WS-B2's future appworld work should consume this example.
- **Env seam** (D9/D22 amendments) consumed as-is; deskbook already validated it end-to-end.
- **Judge overhaul** (PR #83) changes fidelity scales — leg-1 fidelity reported on the
  then-current mainline judge; gate accuracy is judge-independent by construction (that's a
  selling point).
- **FF3 divergence runner** reused if landed; else example-local.

## Risks

- AppWorld version instability → pin + record the pin in the README and corpus metadata.
- Determinism not textually guaranteed → P0 test replays a recorded episode twice and diffs
  DB state; if nondeterminism appears, snapshot/load_state around each step.
- Gate classes unbalanced in natural traces → stratified adversarial instructions (P1) and
  per-family reporting rather than a single blended number.
- WM gate accuracy embarrassingly low → still a launch: the number IS the neural-OS argument,
  and the per-family breakout keeps it constructive.
