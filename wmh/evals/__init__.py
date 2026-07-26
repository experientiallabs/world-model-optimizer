"""World-model evaluation: one interface, an open-loop and a closed-loop implementation.

- `base` — the general `Evaluation`/`EvalResult` interface: mode-specific inputs are bound at
  construction; `run()` returns a report with a one-line `summary()` and a `headline` score.
- `open_loop` — teacher-forced replay of held-out trace steps, scored for per-step reconstruction
  fidelity (the default `wmh eval` mode).
- `closed_loop` — a live agent runs tasks with the world model as its environment, gold-judged for
  end-to-end task success over k=3 passes (`wmh eval --mode closed-loop`).
- `agreement` — compare two closed-loop reports task-by-task (e.g. simulated vs real): the
  outcome-agreement validity check.
- `gold` / `tasks` — the gold-assertion judge and the task specs closed-loop eval scores against.
"""

from wmh.evals.agreement import AgreementReport, compute_agreement
from wmh.evals.base import EvalResult, Evaluation
from wmh.evals.closed_loop import (
    ClosedLoopEval,
    ClosedLoopReport,
    TaskOutcome,
    WorldModelEnvironment,
    WorldModelSource,
    evaluate_closed_loop,
    evaluate_with_env,
)
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.grid import GridCell, GridResult, ModelSpec, merge_results, run_grid
from wmh.evals.open_loop import EvalReport, OpenLoopEval, evaluate_files
from wmh.evals.tasks import TaskSpec, load_tasks

__all__ = [
    "AgreementReport",
    "ClosedLoopEval",
    "ClosedLoopReport",
    "EvalReport",
    "EvalResult",
    "Evaluation",
    "GoldJudge",
    "GoldVerdict",
    "GridCell",
    "GridResult",
    "ModelSpec",
    "OpenLoopEval",
    "TaskOutcome",
    "TaskSpec",
    "WorldModelEnvironment",
    "WorldModelSource",
    "compute_agreement",
    "evaluate_closed_loop",
    "evaluate_files",
    "evaluate_with_env",
    "load_tasks",
    "merge_results",
    "run_grid",
]
