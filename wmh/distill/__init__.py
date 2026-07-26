"""Distillation optimizer mode, on-policy and off-policy.

Trains a Tinker LoRA student from rollouts harbor's own terminus-2 agent
produces on harbor tasks (TerminalBench-2). On-policy, the student samples
through Tinker, the teacher scores those exact tokens via compute_logprobs,
and the reverse-KL advantages feed the advantage-weighted loss `train.loss`
names (`importance_sampling` or `ppo`). Off-policy (`[offpolicy]`), the
TEACHER samples the trajectories and the student trains hard-target
cross entropy over them for a resumable schedule of epochs and minibatches
(`wmh.distill.offpolicy`).
The public surface is deliberately minimal for now: the per-run TOML config
model and its loader.
"""

from wmh.distill.config import DistillConfig, load_distill_config

__all__ = ["DistillConfig", "load_distill_config"]
