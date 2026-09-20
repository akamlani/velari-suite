"""Tests for velari_ai.ai.evals.scoring."""
import numpy as np
import pytest


class TestEditDistance:
    def test_string_substitutions_and_insertions_counted(self):
        from velari_ai.ai.evals.scoring import edit_distance
        assert edit_distance("kitten", "sitting") == 3

    def test_non_string_sequences_supported(self):
        from velari_ai.ai.evals.scoring import edit_distance
        assert edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1


class TestCrossEntropy:
    def test_single_half_probability_is_ln2_nats(self):
        from velari_ai.ai.evals.scoring import cross_entropy
        assert cross_entropy([0.5]) == pytest.approx(np.log(2))

    def test_logprobs_and_probabilities_agree(self):
        from velari_ai.ai.evals.scoring import cross_entropy
        probs = [0.95, 0.73, 0.30, 0.98]
        assert cross_entropy(np.log(probs), is_logprob=True) == pytest.approx(cross_entropy(probs))

    @pytest.mark.parametrize("values", [[], [0.5, 0.0], [0.5, -0.1]])
    def test_invalid_probabilities_raise_valueerror(self, values):
        from velari_ai.ai.evals.scoring import cross_entropy
        with pytest.raises(ValueError):
            cross_entropy(values)

    def test_positive_logprob_input_is_not_rejected_as_invalid_probability(self):
        from velari_ai.ai.evals.scoring import cross_entropy
        assert cross_entropy([-0.5, -1.5], is_logprob=True) == pytest.approx(1.0)


class TestPerplexity:
    def test_uniform_over_two_tokens_is_two(self):
        from velari_ai.ai.evals.scoring import perplexity
        assert perplexity([0.5, 0.5, 0.5]) == pytest.approx(2.0)

    def test_fully_certain_tokens_is_one(self):
        from velari_ai.ai.evals.scoring import perplexity
        assert perplexity([1.0, 1.0]) == pytest.approx(1.0)

    def test_matches_log2_reference_formula(self):
        from velari_ai.ai.evals.scoring import perplexity
        probs = np.array([0.95, 0.73, 0.30, 0.98])
        assert perplexity(probs) == pytest.approx(2 ** (-np.mean(np.log2(probs))))

    def test_logprobs_and_probabilities_agree(self):
        from velari_ai.ai.evals.scoring import perplexity
        probs = [0.95, 0.73, 0.30, 0.98]
        assert perplexity(np.log(probs), is_logprob=True) == pytest.approx(perplexity(probs))
