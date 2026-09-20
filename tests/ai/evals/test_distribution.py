"""Tests for velari_ai.ai.evals.distribution."""
import numpy as np
import pytest
from scipy.stats import entropy


P = [0.70, 0.20, 0.05, 0.05]
Q = [0.55, 0.30, 0.10, 0.05]


class TestCrossEntropyDist:
    @pytest.mark.parametrize("base, expected", [(2.0, 1.0), (np.e, np.log(2))])
    def test_identical_fair_coin_equals_its_entropy_in_given_base(self, base, expected):
        from velari_ai.ai.evals.distribution import cross_entropy_dist
        assert cross_entropy_dist([0.5, 0.5], [0.5, 0.5], base=base) == pytest.approx(expected)

    def test_one_hot_target_matches_token_cross_entropy(self):
        from velari_ai.ai.evals.distribution import cross_entropy_dist
        from velari_ai.ai.evals.scoring import cross_entropy
        assert cross_entropy_dist([0, 1, 0, 0], Q, base=np.e) == pytest.approx(cross_entropy([Q[1]]))

    def test_2d_inputs_return_one_value_per_row(self):
        from velari_ai.ai.evals.distribution import cross_entropy_dist
        scores = cross_entropy_dist(np.stack([P, P]), np.stack([Q, P]))
        assert scores.shape == (2,)
        assert scores[0] == pytest.approx(cross_entropy_dist(P, Q))

    def test_zero_target_probability_term_is_ignored(self):
        from velari_ai.ai.evals.distribution import cross_entropy_dist
        assert cross_entropy_dist([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_zero_prediction_where_target_positive_is_large_but_finite(self):
        from velari_ai.ai.evals.distribution import cross_entropy_dist
        assert np.isfinite(cross_entropy_dist([0.5, 0.5], [1.0, 0.0]))

    @pytest.mark.parametrize("p, q", [
        ([], []),
        ([0.5, 0.5], [1.0]),
        ([0.5, 0.5], [0.3, 0.3]),
        ([0.5, 0.6], [0.5, 0.5]),
    ])
    def test_invalid_inputs_raise_valueerror(self, p, q):
        from velari_ai.ai.evals.distribution import cross_entropy_dist
        with pytest.raises(ValueError):
            cross_entropy_dist(p, q)


class TestPerplexityDist:
    def test_identical_uniform_distribution_equals_number_of_outcomes(self):
        from velari_ai.ai.evals.distribution import perplexity_dist
        assert perplexity_dist([0.25] * 4, [0.25] * 4) == pytest.approx(4.0)

    def test_equals_two_to_the_bits_cross_entropy(self):
        from velari_ai.ai.evals.distribution import cross_entropy_dist, perplexity_dist
        assert perplexity_dist(P, Q) == pytest.approx(2 ** cross_entropy_dist(P, Q))


class TestKlDivergence:
    def test_identical_distributions_is_zero(self):
        from velari_ai.ai.evals.distribution import kl_divergence
        assert kl_divergence(P, P) == pytest.approx(0.0)

    def test_matches_scipy_entropy_in_bits(self):
        from velari_ai.ai.evals.distribution import kl_divergence
        assert kl_divergence(P, Q) == pytest.approx(entropy(P, Q, base=2))
