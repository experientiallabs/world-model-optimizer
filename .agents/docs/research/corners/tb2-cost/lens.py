"""The TB2-cost corner's lens spec: the terminal-tasks dataset through the shared runner.

Declarative only (charter Amendment: one shared runner, corners hold a lens spec and
findings prose). The dataset is the PRODUCT pipeline's arm matrices, preserved per arm in
the grid dir layout (`wmo optimize model` overwrites `<model>/optimize/matrix.json` per
arm, a filed product finding). Workload relabel is binding: this corpus is the org's HF
terminal-tasks bundle, NOT Terminal-Bench 2 episodes (see README.md here).

No dial-curve figure: the fitted identity policy routes 0% away from its fallback, so
there are no routed detents worth replaying (the runner still reports routed rungs as
pending/notes, honestly, if a policy.json ever lands beside the matrices).
"""

from build_corners import FigureSpec, LensSpec
from data import main_checkout

LENS = LensSpec(
    name="tb2-cost",
    corner_dir="tb2-cost",
    dataset_root=str(main_checkout() / ".wmo" / "jt" / "tb2cost"),
    dataset_label="terminal-tasks",
    judge_label="terminal-tasks build rubric (opus-4-8)",
    figures=(
        FigureSpec(kind="savings_frontier", filename="savings_vs_fable5.png"),
        FigureSpec(kind="cost_per_task", filename="effective_cost_per_task.png"),
        # The Silen three-stage directive (DECISIONS 2026-07-28); added by the cost chat
        # under extend-never-fork, flagged in DECISIONS for the tb2 chat's review.
        FigureSpec(
            kind="three_stage",
            filename="three_stage_terminal.png",
            params={
                "constant_routed": (
                    "fit fallback sonnet-5, routes 0.0% away (the correct learned "
                    "policy is a constant)"
                ),
                "distill_note": (
                    "terminal-tasks: distillation skipped by\n"
                    "directive (a distilled model already exists\n"
                    "on HF from the earlier TB2 work)."
                ),
            },
        ),
    ),
)
