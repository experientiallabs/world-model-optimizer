"""The COST-MAX corner's lens spec: which figures, rendered by common/build_corners.py.

Declarative only (charter Amendment: corners hold a lens spec and findings prose; the one
shared runner does every computation). Figure kinds live in the runner's registry; a new
need here means extending the runner, never a standalone script.
"""

from build_corners import FigureSpec, LensSpec

LENS = LensSpec(
    name="cost",
    corner_dir="cost",
    figures=(
        FigureSpec(kind="savings_frontier", filename="savings_vs_fable5.png"),
        FigureSpec(kind="cost_per_task", filename="effective_cost_per_task.png"),
        FigureSpec(kind="dial_curve", filename="dial_cost_curve.png"),
        FigureSpec(
            kind="training_stage",
            filename="training_stage_cost_lens.png",
            params={"lens": "cost"},
        ),
        FigureSpec(
            kind="three_stage",
            filename="three_stage_tau.png",
            params={
                "distill_note": (
                    "tau: cycle 1's gate REJECTED the warmup adapter\n"
                    "(no teacher headroom, p=0.45 at n=60); no student\n"
                    "was promoted, and distillation is not pursued\n"
                    "(Silen 2026-07-28). The refusals ARE the mechanism."
                )
            },
        ),
        FigureSpec(
            kind="three_stage_ours9",
            filename="three_stage_ours9.png",
            params={
                "mix": "37.5% gpt-5.5 / 33% sonnet-5 / 23% fable-5 / 5% opus-4-8 "
                "(frontier-pinned, routing lane)"
            },
        ),
    ),
)
