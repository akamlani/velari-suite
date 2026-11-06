"""Tests for velari_ai.integrations.sklearn.scoring."""
import numpy as np
import pytest


@pytest.mark.parametrize("a, b", [
    ([1, 0], [0, 1]),
    (np.array([1, 0]), np.array([0, 1])),
])
def test_metric_cosine_sim_1d_inputs_return_1x1_matrix(a, b):
    from velari_ai.integrations.sklearn.scoring import metric_cosine_sim
    scores = metric_cosine_sim(a, b)
    assert scores.shape == (1, 1)
    assert scores[0, 0] == pytest.approx(0.0)


def test_metric_cosine_sim_1d_query_against_2d_corpus():
    from velari_ai.integrations.sklearn.scoring import metric_cosine_sim
    scores = metric_cosine_sim([1, 0], [[1, 0], [0, 1]])
    assert scores.shape == (1, 2)
    assert scores[0] == pytest.approx([1.0, 0.0])


def test_metric_cosine_sim_without_b_is_self_similarity():
    from velari_ai.integrations.sklearn.scoring import metric_cosine_sim
    scores = metric_cosine_sim([[1, 0], [1, 1]])
    assert scores.shape == (2, 2)
    assert np.diag(scores) == pytest.approx([1.0, 1.0])


def test_metric_cosine_sim_3d_input_flattens_trailing_axes():
    from velari_ai.integrations.sklearn.scoring import metric_cosine_sim
    scores = metric_cosine_sim(np.ones((3, 2, 2)))
    assert scores.shape == (3, 3)


def test_scorer_clustering_returns_the_five_metric_groups():
    from velari_ai.integrations.sklearn.scoring import scorer_clustering
    scores = scorer_clustering(["billing", "billing", "returns", "returns"], [1, 1, 0, 0])
    assert list(scores) == ["pairwise", "information", "entropy", "matching", "outliers"]


def test_scorer_clustering_relabeled_identical_grouping_scores_one():
    from velari_ai.integrations.sklearn.scoring import scorer_clustering
    scores = scorer_clustering(["billing", "billing", "returns", "returns"], [1, 1, 0, 0])
    flat   = {metric: value for group in scores.values() for metric, value in group.items()}
    assert flat.pop("noise_fraction") == 0.0
    assert list(flat.values()) == pytest.approx([1.0] * 9)


def test_scorer_clustering_singleton_clusters_are_homogeneous_but_incomplete():
    from velari_ai.integrations.sklearn.scoring import scorer_clustering
    group = scorer_clustering(["billing", "billing", "returns", "returns"], [0, 1, 2, 3])["entropy"]
    assert group["homogeneity"] == pytest.approx(1.0)
    assert group["completeness"] < 1.0


def test_scorer_clustering_noise_fraction_counts_minus_one_labels():
    from velari_ai.integrations.sklearn.scoring import scorer_clustering
    assert scorer_clustering(["a", "a", "b", "b"], [0, -1, 1, -1])["outliers"]["noise_fraction"] == 0.5
