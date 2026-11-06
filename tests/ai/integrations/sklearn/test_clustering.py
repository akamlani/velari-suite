"""Tests for velari_ai.integrations.sklearn.clustering."""
import numpy as np
import pytest
from omegaconf import OmegaConf
from sklearn.datasets import make_blobs


@pytest.fixture
def X():
    return make_blobs(n_samples=60, n_features=8, centers=3, random_state=0)[0]


def test_hdbscan_fit_predict_blobs_finds_three_clusters(X):
    from velari_ai.integrations.sklearn.clustering import HDBSCANDriver
    driver = HDBSCANDriver(min_cluster_size=5)
    assert driver.fit_predict(X).shape == (60,)
    assert driver.num_clusters == 3


def test_hdbscan_predict_returns_label_per_row(X):
    from velari_ai.integrations.sklearn.clustering import HDBSCANDriver
    driver = HDBSCANDriver(min_cluster_size=5).fit(X)
    assert driver.predict(X[:5]).shape == (5,)


def test_hdbscan_predict_proba_has_column_per_cluster(X):
    from velari_ai.integrations.sklearn.clustering import HDBSCANDriver
    driver = HDBSCANDriver(min_cluster_size=5).fit(X)
    assert driver.predict_proba(X[:5]).shape == (5, 3)


def test_hdbscan_predict_proba_without_clusters_has_zero_columns():
    from velari_ai.integrations.sklearn.clustering import HDBSCANDriver
    noise = np.random.default_rng(0).normal(size=(8, 4))
    driver = HDBSCANDriver(min_cluster_size=5).fit(noise)
    assert driver.num_clusters == 0
    assert driver.predict_proba(noise[:2]).shape == (2, 0)


def test_hdbscan_fit_predict_small_sample_clamps_min_cluster_size():
    from velari_ai.integrations.sklearn.clustering import HDBSCANDriver
    driver = HDBSCANDriver(min_cluster_size=10)
    driver.fit_predict(np.asarray(make_blobs(n_samples=4, n_features=3, random_state=0)[0]))
    assert driver.get_parameters()["min_cluster_size"] == 4


def test_kmeans_fit_predict_small_sample_clamps_n_clusters():
    from velari_ai.integrations.sklearn.clustering import KMeansDriver
    driver = KMeansDriver(n_clusters=8)
    driver.fit_predict(np.asarray(make_blobs(n_samples=6, n_features=3, random_state=0)[0]))
    assert driver.get_parameters()["n_clusters"] == 3


def test_kmeans_predict_proba_rows_sum_to_one(X):
    from velari_ai.integrations.sklearn.clustering import KMeansDriver
    driver = KMeansDriver(n_clusters=3).fit(X)
    assert driver.predict_proba(X[:5]).sum(axis=1) == pytest.approx(1.0)


@pytest.mark.parametrize("overrides, expected", [({}, 4), ({"min_cluster_size": 6}, 6)])
def test_from_config_overrides_win_over_config_values(overrides, expected):
    from velari_ai.integrations.sklearn.clustering import HDBSCANDriver
    driver = HDBSCANDriver.from_config(OmegaConf.create({"min_cluster_size": 4}), seed=7, **overrides)
    assert driver.get_parameters()["min_cluster_size"] == expected


@pytest.mark.parametrize("driver_name", ["HDBSCANDriver", "KMeansDriver"])
def test_predict_before_fit_raises_runtimeerror(X, driver_name):
    from velari_ai.integrations.sklearn import clustering
    with pytest.raises(RuntimeError):
        getattr(clustering, driver_name)().predict(X)


def test_hdbscan_to_frame_with_labels_adds_score_columns(X):
    from velari_ai.integrations.sklearn.clustering import HDBSCANDriver
    df_clusters = HDBSCANDriver(min_cluster_size=5).to_frame(X, labels=["a"] * 60)
    assert df_clusters.columns.tolist() == ["cluster", "label", "probability", "outlier_score"]


def test_kmeans_to_frame_without_labels_has_only_cluster_column(X):
    from velari_ai.integrations.sklearn.clustering import KMeansDriver
    assert KMeansDriver(n_clusters=3).to_frame(X).columns.tolist() == ["cluster"]
