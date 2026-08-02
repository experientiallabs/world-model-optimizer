# Coding router scientific stop audit

Status: terminal external development conclusion on 2026-08-02. The 320-task external
confirmation and every DeepSWE outcome remain sealed.

## Decision

Stop adding pre-call, latency-neutral router families under the current frozen constraints. The
remaining objective is not scientifically credible without either information revealed during an
agent trajectory or a materially different outcome source that measures the same model by
reasoning-effort arms.

This is a negative development decision, not evidence that routing is impossible in general. It
is specific to a repository-disjoint transfer setting where a route must be chosen before the
first model call, no inference-time LLM may help, no fitted numeric state may persist, and the
policy must retain at least 95 percent quality while saving at least 40 percent.

No confirmation route is frozen. No confirmation or DeepSWE outcome is opened. Rough cumulative
spend remains USD 4,135.54607635.

## Evidence audit

The audited 649-task graded SWE-rebench v47 development matrix has six effort-specific arms and
96.4 percent whole-task coverage. It contains real conditional headroom: every pair oracle that
includes `sol-max` saves 56.9 to 61.4 percent while improving mean graded reward. The failure is
therefore learnability from allowed pre-call inputs, not absence of complementary outcomes.

The completed external families cover the credible information sources available under the
frozen route contract:

| Family | Distinct information tested | External result |
| --- | --- | --- |
| Character and semantic guarded kNN | lexical and pretrained semantic task proximity | quality-safe savings only 9.7 to 13.4 percent |
| Direct and pooled uplift | supervised task by arm reward differences | no externally confirmed signal; one confirmation gain matched a shuffled control |
| Public trace difficulty | generic SWE difficulty learned from 24,100 trajectories | reproducible difficulty signal, but only 6.4 percent savings near the quality floor |
| Public trace interaction | model-specific success differences from multi-model traces | negative repository-held-out correlation and inadequate source overlap |
| Graded IRT plus KL robustness | ordered effort ability, task difficulty, discrimination, and repository shift | 4,000 fits, zero eligible policies, at most 14.0 percent savings at the quality floor |
| Workload budget allocation | task-text contexts and shared budget optimization | best-quality point retained 84.42 percent and saved 38.16 percent |
| Repository tree and issue localization | pre-call codebase structure and path localization | source identity capped coverage at 43.297 percent before fitting |

Repeated-attempt source screens also show why more generic coding traces do not repair the
problem. BigCodeBench held-out oracle headroom was 0.01656. A 60,000-cell SWE-smith expert pool
had 0.00494 held-out oracle headroom. Codeforces produced effort effects, but its learned route
failed the matched task-blind and grouped uncertainty controls. These sources either lack stable
arm complementarity or do not transfer task-conditioned effort uplift.

## Literature boundary

The current primary literature points to information outside the frozen route contract rather
than another credible pre-call representation.

SWE-Router conditions on a cheap model's partial trajectory and gives a formal reason that this
can improve over prompt-only routing: `https://arxiv.org/abs/2607.00053`. That approach adds model
execution before the route decision and is excluded here.

TwinRouterBench trains on router-visible step prefixes containing tool results, shell logs, and
partial edits, then validates by live step-level execution:
`https://arxiv.org/abs/2605.18859`. Those prefixes do not exist before the first call, and a
dynamic fitted policy cannot be represented by the required label-free task route manifest.

WISERouter is the strongest distinct workload-budget proposal found in the current literature:
`https://arxiv.org/abs/2607.23765`. Its frozen external implementation already failed the primary
quality and savings gates. IRT-Router, RACER, and POLLINATOR motivated the completed IRT, robust,
and graph variants, which also failed development.

## Why another prompt-only fit is not justified

Changing the prompt encoder, learner, threshold grid, or guard after observing these failures
would be repeated tuning of the same information set. The experiment has already crossed sparse
and dense lexical features, pretrained semantic features, linear and nonlinear supervised heads,
local similarity, clustering, latent item-response structure, robust allocation, and independent
trace priors. Every positive-looking point was rejected by savings, quality, grouped uncertainty,
matched-blind, null, or source-coverage evidence.

The remaining public trajectory datasets are scientifically useful only if trajectory prefixes
become legal inference inputs or if they contain dense outcomes for the same model by reasoning
effort. Generic successful trajectories provide more task difficulty supervision, which was
already shown not to identify the arm interaction needed for this objective. Repeating that lane
would not be a new hypothesis.

## Conditions that would reopen the goal

One of these material changes would justify a new preregistration:

1. Allow a bounded cheap exploration prefix and count its cost and latency end to end.
2. Allow a fitted local step-level router to remain available at inference.
3. Provide a new external repository-disjoint corpus with dense graded outcomes for the same
   model by reasoning-effort arms.
4. Relax the 95 percent quality or 40 percent savings gate.

Without one of those changes, external confirmation and the single DeepSWE transfer stay sealed.
This preserves the only remaining confirmatory evidence instead of spending it on another member
of an exhausted pre-call family.
