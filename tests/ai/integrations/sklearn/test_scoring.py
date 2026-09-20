"""Tests for velari_ai.integrations.sklearn.scoring."""
import numpy as np
import pytest


@pytest.mark.parametrize("a, b", [
    ([1, 0], [0, 1]),
    (np.array([1, 0]), np.array([0, 1])),
])
def test_cosine_scores_1d_inputs_return_1x1_matrix(a, b):
    from velari_ai.integrations.sklearn.scoring import cosine_scores
    scores = cosine_scores(a, b)
    assert scores.shape == (1, 1)
    assert scores[0, 0] == pytest.approx(0.0)


def test_cosine_scores_1d_query_against_2d_corpus():
    from velari_ai.integrations.sklearn.scoring import cosine_scores
    scores = cosine_scores([1, 0], [[1, 0], [0, 1]])
    assert scores.shape == (1, 2)
    assert scores[0] == pytest.approx([1.0, 0.0])


def test_cosine_scores_without_b_is_self_similarity():
    from velari_ai.integrations.sklearn.scoring import cosine_scores
    scores = cosine_scores([[1, 0], [1, 1]])
    assert scores.shape == (2, 2)
    assert np.diag(scores) == pytest.approx([1.0, 1.0])


def test_cosine_scores_3d_input_flattens_trailing_axes():
    from velari_ai.integrations.sklearn.scoring import cosine_scores
    scores = cosine_scores(np.ones((3, 2, 2)))
    assert scores.shape == (3, 3)
