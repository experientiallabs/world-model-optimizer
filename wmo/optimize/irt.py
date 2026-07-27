"""IRT routing head: IrtNet's 2PL model, numpy-only (train AND serve).

Replicates the reference (arXiv 2510.00844, spec in .agents/docs/reference/irtnet-spec.md):
P(correct | model m, query q) = sigmoid(alpha_q . theta_m - beta_q), where theta_m is a learned
per-model ability vector and a small head maps the frozen query embedding to the per-query
discrimination vector alpha_q and difficulty scalar beta_q. Their published ablation shows the
plain-MLP head (this module) keeps most of the win over Avengers-Pro (64.0 vs 62.1; the full
39-expert dense MoE reaches 67.4) at ~1/40th the parameters, which is the right size for
per-endpoint data volumes. Deliberate deltas from the reference, in one place:
(1) MLP head, no MoE (configurable later); (2) query embeddings come from the policy's
EmbedderSpec (hashing / azure), not mpnet; (3) training is hand-rolled full-batch Adam in
numpy (BCE loss, the reference's lr/weight-decay defaults) so wmo needs no torch anywhere -
guarded by a finite-difference gradient check in the tests; (4) weights persist INLINE on the
policy artifact (self-contained policy files).

Routing note from the reference: beta_q is constant across models, so pure-accuracy routing
reduces to argmax(alpha_q . theta_m); beta matters once probabilities feed a cost knob or
abstention, so predict() returns calibrated-ish P, not just the argmax.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from wmo.optimize.outcomes import OutcomeMatrix

_LR = 1e-3
_WEIGHT_DECAY = 1e-4


class IrtHead(BaseModel):
    """The fitted 2PL head: everything serving needs, JSON-serializable."""

    models: list[str]  # theta row order (the reference's model_map, persisted)
    theta: list[list[float]]  # [M, dim] per-model ability vectors
    w1: list[list[float]]  # [hidden, D] query MLP
    b1: list[float]
    wa: list[list[float]]  # [dim, hidden] discrimination head
    ba: list[float]
    wb: list[float]  # [hidden] difficulty head (dot + scalar bias)
    bb: float
    pairs_trained: int = 0
    final_loss: float = 0.0

    def predict(self, query: np.ndarray) -> np.ndarray:
        """P(correct) per model (theta-row order) for one embedded query."""
        hidden = np.maximum(np.asarray(self.w1) @ query + np.asarray(self.b1), 0.0)
        alpha = np.asarray(self.wa) @ hidden + np.asarray(self.ba)
        beta = float(np.asarray(self.wb) @ hidden + self.bb)
        logits = np.asarray(self.theta) @ alpha - beta
        return 1.0 / (1.0 + np.exp(-logits))


class _Params:
    """Mutable training view (numpy arrays); exported to IrtHead at the end."""

    def __init__(
        self, rng: np.random.Generator, n_models: int, d_in: int, hidden: int, dim: int
    ) -> None:
        scale = lambda fan_in: 1.0 / np.sqrt(fan_in)  # noqa: E731 - local init helper
        self.theta = rng.normal(0, scale(dim), (n_models, dim))
        self.w1 = rng.normal(0, scale(d_in), (hidden, d_in))
        self.b1 = np.zeros(hidden)
        self.wa = rng.normal(0, scale(hidden), (dim, hidden))
        self.ba = np.zeros(dim)
        self.wb = rng.normal(0, scale(hidden), hidden)
        self.bb = 0.0

    def tensors(self) -> list[np.ndarray]:
        return [self.theta, self.w1, self.b1, self.wa, self.ba, self.wb]


def _forward_loss_grads(
    params: _Params, queries: np.ndarray, model_index: np.ndarray, labels: np.ndarray
) -> tuple[float, list[np.ndarray], float]:
    """BCE loss + analytic gradients over all (query, model, label) pairs at once.

    `queries` is [P, D] (one row per PAIR, duplicated per model), `model_index` [P] into theta,
    `labels` [P] in {0,1}. Returns (loss, grads aligned with params.tensors(), grad_bb).
    """
    pairs = len(labels)
    pre = queries @ params.w1.T + params.b1  # [P, H]
    hidden = np.maximum(pre, 0.0)
    alpha = hidden @ params.wa.T + params.ba  # [P, dim]
    beta = hidden @ params.wb + params.bb  # [P]
    theta_rows = params.theta[model_index]  # [P, dim]
    logits = np.sum(alpha * theta_rows, axis=1) - beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    eps = 1e-12
    loss = float(-np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)))

    dlogit = (probs - labels) / pairs  # [P]
    # theta grads scatter-add per model row.
    d_theta = np.zeros_like(params.theta)
    np.add.at(d_theta, model_index, dlogit[:, None] * alpha)
    d_alpha = dlogit[:, None] * theta_rows  # [P, dim]
    d_beta = -dlogit  # [P]
    d_hidden = d_alpha @ params.wa + d_beta[:, None] * params.wb  # [P, H]
    d_pre = d_hidden * (pre > 0)
    grads = [
        d_theta,
        d_pre.T @ queries,  # w1
        d_pre.sum(axis=0),  # b1
        d_alpha.T @ hidden,  # wa
        d_alpha.sum(axis=0),  # ba
        hidden.T @ d_beta,  # wb
    ]
    return loss, grads, float(d_beta.sum())


def fit_irt_head(
    matrix: OutcomeMatrix,
    *,
    scenario_ids: list[str],
    embeddings: np.ndarray,
    seed: int = 42,
    epochs: int = 200,
    hidden: int = 128,
    dim: int = 64,
    lr: float = _LR,
    weight_decay: float = _WEIGHT_DECAY,
) -> IrtHead:
    """Fit the 2PL head on a matrix's scored outcomes (full-batch Adam, deterministic in seed).

    `scenario_ids`/`embeddings` are aligned (the caller embeds tasks once); graded rewards are
    used as-is as targets (binary on exact-match data, soft on judge-scored data - the DARS-
    friendly generalization). Unscored outcomes are excluded, never zero-filled.
    """
    row_of = {sid: index for index, sid in enumerate(scenario_ids)}
    model_names = [entry.name for entry in matrix.pool]
    model_row = {name: index for index, name in enumerate(model_names)}
    query_rows, model_index, labels = [], [], []
    for outcome in matrix.outcomes:
        if outcome.reward is None or outcome.scenario_id not in row_of:
            continue
        query_rows.append(row_of[outcome.scenario_id])
        model_index.append(model_row[outcome.model])
        labels.append(outcome.reward)
    if not labels:
        raise ValueError("no scored outcomes to fit the IRT head on")
    queries = np.asarray(embeddings, dtype=np.float64)[query_rows]
    model_idx = np.asarray(model_index)
    label_arr = np.asarray(labels, dtype=np.float64)

    rng = np.random.default_rng(seed)
    params = _Params(rng, len(model_names), queries.shape[1], hidden, dim)
    moments = [(np.zeros_like(t), np.zeros_like(t)) for t in params.tensors()]
    m_bb = v_bb = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    loss = 0.0
    for step in range(1, epochs + 1):
        loss, grads, grad_bb = _forward_loss_grads(params, queries, model_idx, label_arr)
        tensors = params.tensors()
        for index, (tensor, grad) in enumerate(zip(tensors, grads, strict=True)):
            grad = grad + weight_decay * tensor
            m, v = moments[index]
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**step)
            v_hat = v / (1 - beta2**step)
            tensor -= lr * m_hat / (np.sqrt(v_hat) + eps)
        m_bb = beta1 * m_bb + (1 - beta1) * grad_bb
        v_bb = beta2 * v_bb + (1 - beta2) * grad_bb**2
        params.bb -= lr * (m_bb / (1 - beta1**step)) / (np.sqrt(v_bb / (1 - beta2**step)) + eps)

    return IrtHead(
        models=model_names,
        theta=params.theta.tolist(),
        w1=params.w1.tolist(),
        b1=params.b1.tolist(),
        wa=params.wa.tolist(),
        ba=params.ba.tolist(),
        wb=params.wb.tolist(),
        bb=params.bb,
        pairs_trained=len(labels),
        final_loss=round(loss, 6),
    )


def irt_gradient_check(seed: int = 0) -> float:
    """Max relative error between analytic and finite-difference gradients (test hook)."""
    rng = np.random.default_rng(seed)
    params = _Params(rng, n_models=3, d_in=5, hidden=4, dim=3)
    queries = rng.normal(size=(6, 5))
    model_index = np.asarray([0, 1, 2, 0, 1, 2])
    labels = np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    _, grads, grad_bb = _forward_loss_grads(params, queries, model_index, labels)
    worst = 0.0
    h = 1e-6
    for tensor, grad in zip(params.tensors(), grads, strict=True):
        flat = tensor.ravel()
        for k in rng.choice(flat.size, size=min(10, flat.size), replace=False):
            original = flat[k]
            flat[k] = original + h
            up, _, _ = _forward_loss_grads(params, queries, model_index, labels)
            flat[k] = original - h
            down, _, _ = _forward_loss_grads(params, queries, model_index, labels)
            flat[k] = original
            numeric = (up - down) / (2 * h)
            denom = max(abs(numeric), abs(grad.ravel()[k]), 1e-8)
            worst = max(worst, abs(numeric - grad.ravel()[k]) / denom)
    params.bb += h
    up, _, _ = _forward_loss_grads(params, queries, model_index, labels)
    params.bb -= 2 * h
    down, _, _ = _forward_loss_grads(params, queries, model_index, labels)
    params.bb += h
    numeric = (up - down) / (2 * h)
    worst = max(worst, abs(numeric - grad_bb) / max(abs(numeric), abs(grad_bb), 1e-8))
    return worst
