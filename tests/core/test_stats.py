"""Tests for velari_core.core.statistics.stats."""
import warnings
import numpy as np
import pandas as pd
import pytest


class TestCalcZScore:
    def test_matches_scipy_zscore_for_population_std(self):
        from scipy.stats import zscore
        from velari_core.core.statistics.stats import calc_z_score
        values = [1.2, 1.4, 1.1, 1.3, 4.8]
        assert calc_z_score(values) == pytest.approx(zscore(values))

    def test_ddof_1_uses_sample_std(self):
        from scipy.stats import zscore
        from velari_core.core.statistics.stats import calc_z_score
        values = [1.2, 1.4, 1.1, 1.3, 4.8]
        assert calc_z_score(values, ddof=1) == pytest.approx(zscore(values, ddof=1))

    def test_scalar_with_reference_mean_and_std_returns_float(self):
        from velari_core.core.statistics.stats import calc_z_score
        assert calc_z_score(3.0, mean=1.0, std=0.5) == pytest.approx(4.0)

    def test_array_with_reference_mean_and_std_skips_self_statistics(self):
        from velari_core.core.statistics.stats import calc_z_score
        assert calc_z_score(np.array([1.0, 2.0]), mean=1.0, std=0.5) == pytest.approx([0.0, 2.0])

    @pytest.mark.parametrize("values, kwargs", [
        ([2.0, 2.0, 2.0], {}),
        (3.0, {}),
        ([1.0, 2.0], {"mean": 1.0, "std": 0.0}),
    ])
    def test_zero_std_raises_valueerror(self, values, kwargs):
        from velari_core.core.statistics.stats import calc_z_score
        with pytest.raises(ValueError):
            calc_z_score(values, **kwargs)


class TestCalcConfidenceInterval:
    def test_matches_scipy_interval_for_mean_of_values(self):
        from scipy.stats import norm, sem
        from velari_core.core.statistics.stats import calc_confidence_interval
        values = np.array([0.5, -0.2, 1.1, 0.3, 0.9])
        expected = norm.interval(0.95, loc=values.mean(), scale=sem(values))
        assert calc_confidence_interval(values) == pytest.approx(expected)

    def test_higher_confidence_widens_interval(self):
        from velari_core.core.statistics.stats import calc_confidence_interval
        values = [0.5, -0.2, 1.1, 0.3]
        low90, up90 = calc_confidence_interval(values, confidence=0.90)
        low99, up99 = calc_confidence_interval(values, confidence=0.99)
        assert (up99 - low99) > (up90 - low90)

    @pytest.mark.parametrize("values, confidence", [([0.5], 0.95), ([0.5, 0.1], 0.0), ([0.5, 0.1], 1.0), ([0.5, 0.1], 1.5)])
    def test_too_few_values_or_invalid_confidence_raises_valueerror(self, values, confidence):
        from velari_core.core.statistics.stats import calc_confidence_interval
        with pytest.raises(ValueError):
            calc_confidence_interval(values, confidence=confidence)


class TestCalcCosineSimilarity:
    @pytest.mark.parametrize("a, b, expected", [
        ([1, 0], [1, 0], 1.0),
        ([1, 0], [0, 1], 0.0),
        ([1, 0], [-1, 0], -1.0),
        ([0, 0], [1, 1], 0.0),
    ])
    def test_known_vector_pairs_including_zero_vector(self, a, b, expected):
        from velari_core.core.statistics.stats import calc_cosine_similarity
        assert calc_cosine_similarity(np.array(a), np.array(b)) == pytest.approx(expected)

    def test_matrix_of_unit_vectors_is_symmetric_with_unit_diagonal(self):
        from velari_core.core.statistics.stats import calc_cosine_similarity_matrix
        embeddings = np.array([[1.0, 0.0], [0.6, 0.8]])
        sims = calc_cosine_similarity_matrix(embeddings)
        assert sims == pytest.approx(sims.T)
        assert np.diag(sims) == pytest.approx([1.0, 1.0])
        assert sims[0, 1] == pytest.approx(0.6)


class TestCalcPctChange:
    @pytest.mark.parametrize("base, new, expected", [(100, 150, 50.0), (200, 100, -50.0), (1.85, 1.20, -35.135135135135144)])
    def test_increase_and_decrease_signed_correctly(self, base, new, expected):
        from velari_core.core.statistics.stats import calc_pct_change
        assert calc_pct_change(base, new) == pytest.approx(expected)

    def test_zero_base_raises_valueerror(self):
        from velari_core.core.statistics.stats import calc_pct_change
        with pytest.raises(ValueError):
            calc_pct_change(0, 5)


class TestCalcWeightedAvg:
    def test_matches_numpy_average(self):
        from velari_core.core.statistics.stats import calc_weighted_avg
        latencies, tokens = [1.2, 2.5, 0.8], [500.0, 1500.0, 200.0]
        assert calc_weighted_avg(latencies, tokens) == pytest.approx(np.average(latencies, weights=tokens))

    @pytest.mark.parametrize("values, weights", [([1.0, 2.0], [1.0]), ([1.0, 2.0], [0.0, 0.0])])
    def test_length_mismatch_or_zero_weight_sum_raises_valueerror(self, values, weights):
        from velari_core.core.statistics.stats import calc_weighted_avg
        with pytest.raises(ValueError):
            calc_weighted_avg(values, weights)


def _scores(values, group=None):
    return pd.DataFrame({"score": values} if group is None else {"score": values, "variant": group})


class TestCalcColumnStats:
    def test_whole_column_returns_one_row_named_after_the_column_with_the_default_statistics(self):
        from velari_core.core.statistics.stats import calc_column_stats
        df_stats = calc_column_stats(_scores([1.0, 2.0, 3.0, 4.0]), "score")
        assert (df_stats.index.tolist(), df_stats.columns.tolist()) == (["score"], ["count", "mean", "std", "sem", "min", "median", "max", "ci_low", "ci_high"])
        assert (df_stats.loc["score", "count"], df_stats.loc["score", "mean"]) == (4, 2.5)

    def test_groups_come_out_in_sorted_order(self):
        from velari_core.core.statistics.stats import calc_column_stats
        df_stats = calc_column_stats(_scores([1.0, 2.0, 3.0], ["b", "a", "b"]), "score", by="variant").reindex(columns=["count", "mean"])
        assert df_stats.to_dict("index") == {"a": {"count": 1, "mean": 2.0}, "b": {"count": 2, "mean": 2.0}}

    def test_mean_std_and_sem_match_the_standard_library(self):
        import statistics
        from velari_core.core.statistics.stats import calc_column_stats
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        row = calc_column_stats(_scores(values), "score").loc["score", ["mean", "std", "sem"]]
        assert row.tolist() == pytest.approx([statistics.mean(values), statistics.stdev(values), statistics.stdev(values) / len(values) ** 0.5])

    def test_ci_bounds_match_calc_confidence_interval(self):
        from velari_core.core.statistics.stats import calc_column_stats, calc_confidence_interval
        values = [0.5, -0.2, 1.1, 0.3, 0.9]
        row = calc_column_stats(_scores(values), "score").loc["score", ["ci_low", "ci_high"]]
        assert row.tolist() == pytest.approx(calc_confidence_interval(values))

    def test_iqr_mad_and_spread_match_hand_computed_values(self):
        from velari_core.core.statistics.stats import calc_column_stats, iqr, mad, spread
        row = calc_column_stats(_scores([1.0, 2.0, 2.0, 3.0, 9.0]), "score", stats=(iqr, mad, spread)).loc["score", ["iqr", "mad", "spread"]]
        assert row.tolist() == pytest.approx([1.0, 1 / 0.6744897501960817, 8.0])

    @pytest.mark.parametrize("stat, size", [(pd.Series.std, 1), (pd.Series.sem, 1), (pd.Series.skew, 2), (pd.Series.kurt, 3)])
    def test_statistic_is_nan_below_its_minimum_size_without_warning(self, stat, size):
        from velari_core.core.statistics.stats import calc_column_stats
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df_stats = calc_column_stats(_scores([1.0, 2.0, 4.0][:size]), "score", stats=(stat,))
        assert np.isnan(df_stats.loc["score", stat.__name__])

    def test_nan_values_are_ignored_and_counted_in_n_na(self):
        from velari_core.core.statistics.stats import calc_column_stats, n_na
        row = calc_column_stats(_scores([1.0, None, 3.0]), "score", stats=(n_na,)).loc["score", ["count", "n_na", "mean"]]
        assert row.tolist() == [2, 1, 2.0]

    def test_group_with_only_nan_stays_with_zero_count(self):
        from velari_core.core.statistics.stats import calc_column_stats, n_na
        df_stats = calc_column_stats(_scores([1.0, None], ["a", "b"]), "score", by="variant", stats=(n_na,))
        assert (df_stats.loc["b", "count"], df_stats.loc["b", "n_na"], np.isnan(df_stats.loc["b", "mean"])) == (0, 1, True)

    def test_percentile_and_pct_at_name_their_columns_and_compute_their_values(self):
        from velari_core.core.statistics.stats import calc_column_stats, pct_at, percentile
        df_stats = calc_column_stats(_scores([0.0, 50.0, 100.0, 100.0]), "score", stats=(percentile(50), pct_at(0), pct_at(100)))
        assert df_stats.loc["score", ["p50", "pct_at_0", "pct_at_100"]].to_dict() == {"p50": 75.0, "pct_at_0": 25.0, "pct_at_100": 50.0}

    @pytest.mark.parametrize("column, by", [("missing", None), ("score", "missing")])
    def test_missing_column_or_group_column_raises_valueerror(self, column, by):
        from velari_core.core.statistics.stats import calc_column_stats
        with pytest.raises(ValueError):
            calc_column_stats(_scores([1.0, 2.0]), column, by=by)
