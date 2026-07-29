# Quality corner, phase 1: findings

Status 2026-07-27: cycle-1's real-episode leg is measured and plotted; every grid-dependent
number is pending (the three arm matrices are still merging, wall estimate ~15h from launch).
This page states what is known honestly, keeps the negative results, and names the noise
floor everywhere. Statistical conventions: `../common/README.md` (binding on all three
corner chats). Figures and their numbers sidecars: `figures/`.

## 1. The distill rung reads "no measurable effect at this sample size", not a regression

Source: the cycle-1 result note (`.agents/docs/research/tau2-cycle1-result-note.md`, main
checkout) and the 180 per-task rows behind it, recomputed here through the shared paired
stats.

Teacher Qwen3.6-27B 73.3%, student-before 71.7%, student-after 65.0% on the 20 pinned holdout
tasks (k=3, real tau2 episodes, tau2 reward; 7/20 tasks include tau2's NL-assertion judge).
The paired per-task delta of the warmed student against its base is -0.067 with a 95% CI of
[-0.167, +0.033]: the CI spans zero, the sign test over the 7 tasks that moved gives p=0.45,
and 13 of 20 tasks did not move at all. The teacher-student gap was 1.6 points, so there was
no headroom to distill and no resolution to measure it with. The gate refused promotion,
which is the system working, and the honest ladder row is "no measurable effect at this
sample size (teacher had no headroom)". Plotting 65.0 against 71.7 as a regression would be
publishing noise; the shared chart labels the point "not promoted (gate refused: no teacher
headroom)" instead.

Quality-corner consequence: at cycle 1 the distill lever contributes NOTHING measurable to
the quality-max corner, in either direction. The corner's quality story currently rests
entirely on model choice (and, once fits land, guarded routing).

## 2. Small-bank caveat: the routing rung is a cost-at-parity claim, never an accuracy lift

The grid's WM test band has 20 distinct scenarios (the corpus's 1033 traces collapse to 126
distinct task prompts; the deterministic split leaves 20 in test, fit band ~14). The
evidence-volume law from the routing program says routability wants n >= 1000; at n ~ 14 the
guarded router's honest posture is abstention (floor and guard refuse thin evidence). So on
this benchmark the +routing rung's claim is "parity quality at lower cost", and any apparent
quality LIFT from routing at this bank size sits inside the noise floor until proven
otherwise on paired evidence. The quality corner therefore expects its own +routing line to
be flat on the quality axis; the lever's value is measured on the cost axis (the cost
corner's charts, same cells).

## 3. Telecom-skew caveat: the WM panel generalizes to the corpus mix, not to tau2 at large

The WM-simulated ladder inherits the capture corpus's mix, ~85% telecom; the real-episode
leg runs the balanced pinned 20. WM-panel numbers are therefore statements about performance
on that skewed mix. The check between panels is per-scenario paired sign agreement (the
primary sim-to-real statistic per the binding amendment), never a cross-panel mean
comparison, and model-mean rank correlations are quoted descriptive-only if at all.

## 4. What the quality-max corner pays: pending the grid; priors labeled as priors

The corner will be named from measured operating points once matrices and the master's
per-arm fits land (quality-vs-anchor across every measured config, plus effective cost per
completed task and p50 latency reported honestly beside it). Expected shape, stated as
expectation not measurement: dial 0.0, guarded routing, no compression. Prior evidence from
a DIFFERENT dataset, labeled as such: the D-DIAL dial-0.0 anchor measured +1.14 quality
points at -13.9% cost against the best single model on routerbench-ours9; nothing about tau
is claimed from it. This section gets its numbers from `render_quality.py` output when the
grid fills.

## 5. Compaction rungs stay "measured tradeoff, not recommendation"

Binding label until the compaction lane's pre-registered financebench accuracy grid returns
its verdict. Their interim observation is worth carrying: both dumb compression controls
RAISED effective cost per completed task (+21-36%) by deleting load-bearing observations and
lengthening episodes, so the quality cost of compression on tau must be read jointly with
effective cost, never per-token savings. The truncate arm here is a ratio-matched control
(achieved keep 0.5656, aggressiveness 0.33), not a product lever.

## 6. Teacher search: the distill go/no-go, rendered from the same matrices (pending grid)

The teacher-search verdict is becoming a repo function (`wmo.optimize.teacher`, branch
jt/teacher-gate) over the same matrices this corner charts. `figures/teacher-verdict.png`
(renders when the identity arm lands, from pre-retry chunks with completeness labeled) shows
every candidate's PAIRED gain over the cheapest candidate, CI-guarded by the shared verdict
rule: "headroom" needs a CI excluding zero and a mean past the noise floor. The baseline
proxies the student tier until student cells merge (then it becomes the student itself).
Cycle-1 is the motivating negative result: a 1.6-point teacher gap distills nothing, and this
figure is that lesson applied BEFORE spend. Handoff contract in `../common/teacher_view.py`:
when the repo function lands, the corners computation is replaced by rendering its verdict
artifact, and any disagreement between the two is a bug, not a second opinion.

## 7. Negative results kept

- Cycle-1 warmup distillation: gate rejected, adapter unpromoted, see finding 1.
- The +-0.015 to 0.02 noise floor applies to every paired delta on 20-scenario samples;
  `common/stats.py` refuses to call a delta "measurable" unless its CI excludes zero AND its
  mean clears the floor. Cycle-1's -0.067 fails the first test; that verdict is printed on
  the figure itself.
