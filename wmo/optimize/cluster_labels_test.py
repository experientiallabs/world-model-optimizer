"""Tests for cluster labeling."""

from __future__ import annotations

from wmo.optimize.cluster_labels import label_clusters, majority_prefix


def test_ctfidf_labels_are_distinctive() -> None:
    clusters = [
        ["exchange the camera for maximum zoom", "camera zoom specs exchange bird watching"],
        ["cancel order and refund payment", "refund the cancelled order payment method"],
        ["book flight seat upgrade airline", "airline flight cancellation compensation"],
    ]
    labels = label_clusters(clusters)
    assert len(labels) == 3
    assert len(set(labels)) == 3  # distinct clusters get distinct labels
    assert "camera" in labels[0] or "zoom" in labels[0]
    assert "refund" in labels[1] or "order" in labels[1]
    assert "airline" in labels[2] or "flight" in labels[2]


def test_shared_scaffolding_words_do_not_dominate() -> None:
    # Every task shares JSON scaffolding; labels must come from the distinctive payload.
    clusters = [
        ['{"domain": "retail", "reason_for_call": "exchange the camera zoom"}'] * 3,
        ['{"domain": "airline", "reason_for_call": "flight refund compensation"}'] * 3,
    ]
    labels = label_clusters(clusters)
    assert "reason" not in labels[0] and "domain" not in labels[0]
    assert labels[0] != labels[1]


def test_empty_cluster_gets_empty_label() -> None:
    assert label_clusters([[], ["real text about databases sql query"]])[0] == ""


def test_majority_prefix() -> None:
    assert majority_prefix(["mmlu-anatomy:1", "mmlu-anatomy:2", "arc:9"]) == "mmlu-anatomy"
    assert majority_prefix(["deadbeef", "cafebabe"]) == ""
