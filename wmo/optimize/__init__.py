"""Pluggable optimizers (GEPA prompt evolution today) + the LLM judges that score predictions.

The switchable-optimizer interface lives in `wmo.optimize.base`; concrete optimizers implement
its `Optimizer` protocol and return `OptimizeResult`s whose `ArtifactRef`s say what they built.
"""

# Imported for its registration side effect, which is why it is not in __all__: it registers a
# lazily-constructed factory for the `llmlingua2-endpoint` compressor (no credentials read and
# no network at import). Both surfaces that resolve a compressor, `wmo optimize route fit` and
# the serving runtime, reach this package, so a policy naming that id resolves on either path
# without the caller registering anything by hand.
from wmo.optimize import compression_endpoint as compression_endpoint  # noqa: F401
from wmo.optimize import compression_scoped as compression_scoped  # noqa: F401
from wmo.optimize.base import ArtifactRef, OptimizeMetrics, Optimizer, OptimizeResult
from wmo.optimize.gepa import GEPAOptimizer
from wmo.optimize.judge import Judge, JudgeResult, RubricJudge
from wmo.optimize.numeric import NumericJudge
from wmo.optimize.reward import EpisodeRewardJudge, EpisodeScore

__all__ = [
    "ArtifactRef",
    "EpisodeRewardJudge",
    "EpisodeScore",
    "GEPAOptimizer",
    "OptimizeMetrics",
    "OptimizeResult",
    "Optimizer",
    "Judge",
    "JudgeResult",
    "NumericJudge",
    "RubricJudge",
]
