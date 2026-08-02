# External model by reasoning-effort router

Status: frozen before new provider execution on 2026-08-01. DeepSWE outcomes remain sealed.

## Question

Can task-conditioned selection across model and reasoning effort beat both the static frontier and
a matched task-blind mixture on an external execution-scored repository benchmark, strongly enough
to justify one untouched DeepSWE v1.1 transfer?

This is a distinct hypothesis after effort-only task-text routers repeatedly failed external
matched-blind confirmation. It tests whether cross-model complementarity supplies usable signal
that effort-only routing lacks. It does not fit a foundation model or persist fitted numeric router
state.

## External matrix

The source is the frozen 200-task SWE-rebench V2 development cohort from
`coding-model-router-swerebench-effort-20260731.md`. It is repository-disjoint from both the frozen
200-task external confirmation cohort and all label-free DeepSWE tasks. Selection did not access
gold patches, tests, target rewards, or target costs.

The dense development action space is the Cartesian product of:

- `gpt-5.6-luna`, `gpt-5.6-terra`, and `gpt-5.6-sol`;
- `low`, `medium`, `high`, `xhigh`, and `max` reasoning effort.

Every arm gets two attempts per task under the same pinned mini-swe-agent 2.4.5 harness, official
SWE-rebench verifier, turn cap, token cap, and timeout. Existing Luna outcomes are reused exactly.
Terra and Sol run as new isolated E2B matrices. Arm identity is always model plus effort.

## Frozen search and promotion gates

Development uses five repeated repository-grouped five-fold evaluations with seeds 11, 23, 37,
41, and 59. It compares every static arm, selects the strongest static arm on development as the
guard, then evaluates all 14 pairs between that guard and one alternate arm. Dynamic pair policies
route only between the cheaper and more expensive member, which bounds confirmation to two dense
arms while ensuring the quality baseline is directly gradeable. The frozen families are paired
reward-uplift Ridge on signed character 3-to-5-gram hashing
at 512, 2,048, and 8,192 dimensions with alpha 1, 10, or 100, and paired-difference kNN at k 8,
16, 32, or 64. Ridge sends to the expensive arm when predicted expensive-minus-cheap reward
exceeds one of -0.10, 0, 0.02, 0.05, 0.10, or 0.20. kNN uses the same asymmetric paired guard as
WMO with z 0, 0.5, 1, 1.645, or 2 and no absolute similarity floor. Candidate selection minimizes
cost subject to at least 95 percent reward retention versus the strongest fit-selected static arm
on every repeated evaluation. Ties break by frozen family, pair, and parameter order.

Exactly one candidate may be frozen for the untouched external confirmation cohort. Before any
confirmation provider call, refit that configuration on all retained development tasks and freeze
its routes, a task-blind route with exactly the same expensive-arm traffic, and 128 null routes
formed by independently permuting the paired reward-difference labels at repository-group level
before refitting the selected configuration. No null may change the selected pair, family, or
hyperparameters. Promotion requires all of:

1. at least 95 percent retained external reward and at least 40 percent cost savings;
2. positive repository-bootstrap lower bound versus the matched task-blind control;
3. positive repository-bootstrap lower bound versus the best of 128 frozen family nulls;
4. no domination by an eligible static arm at equal or better quality and cost;
5. at least 95 percent whole-task coverage with no selective cell dropping;
6. pre-inference route latency p95 below 5 ms;
7. zero DeepSWE reward or cost access and no persisted fitted numeric router state.

If any gate fails, the result is terminal negative and DeepSWE remains sealed. If all gates pass,
the frozen candidate receives exactly one graded DeepSWE transfer. No target-dependent repair or
rerun is allowed.

## Spend and compute

The user authorized a USD 20,000 hard ceiling and monitors provider usage externally. Exact trace
telemetry is preserved when available; otherwise reporting uses a clearly labeled trace-derived
list-price estimate. All model execution and fitting run on E2B or Azure. The local Mac only
orchestrates, validates small artifacts, and records results. No Modal app or persisted model is
used. The frozen rough cumulative spend before this matrix is USD 1,123.9297378.

## Result

The result is terminal negative on external development. DeepSWE remained sealed and no external
confirmation provider call was launched.

The merged matrix retained 194 of 200 tasks, or 97 percent, with 5,820 dense scientific cells.
Sol completed 194 tasks and excluded 6 whole tasks. Terra completed 195 tasks and excluded 5 whole
tasks. The six-task union was excluded across every model and effort. No scientific cell was
rerun. New trace-derived spend was USD 1,901.17832175, for rough cumulative experiment spend of
USD 3,025.10805955.

The strongest static arm was `sol-max`, with mean reward 0.729381 and mean cost USD 0.959561 per
task. The full 15-arm oracle reached reward 0.840206 at USD 0.257736 per task. Nine of the fourteen
`sol-max` pair oracles could satisfy the frozen point gates. For example, the
`sol-max` plus `luna-low` oracle reached reward 0.744845 at USD 0.505041 per task, 47.37 percent
savings, 102.12 percent quality retention, and 0.208404 matched-blind advantage. The action space
therefore has substantial complementarity.

None of the 1,596 frozen task-text Ridge and kNN candidates passed development. The failure is not
lack of oracle headroom. It is lack of generalizable task-text signal in the tested representation
and estimators. The no-internet E2B fit exited successfully, persisted no fitted numeric state, and
returned only the terminal-negative selection report.
