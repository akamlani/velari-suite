"""Tests for velari_ai.integrations.sklearn.projection."""
import pytest
from sklearn.datasets import make_blobs


@pytest.fixture
def X():
    return make_blobs(n_samples=60, n_features=8, centers=3, random_state=0)[0]


@pytest.mark.parametrize("mode, n_components", [("visualize", 2), ("transform", 10)])
def test_fit_transform_defaults_components_by_mode(X, mode, n_components):
    from velari_ai.integrations.sklearn.projection import ProjectionMode, UMAPProjection
    assert UMAPProjection(mode=ProjectionMode(mode)).fit_transform(X).shape == (60, n_components)


def test_fit_transform_small_sample_clamps_neighbors_and_components():
    from velari_ai.integrations.sklearn.projection import ProjectionMode, UMAPProjection
    projection = UMAPProjection(mode=ProjectionMode.TRANSFORM, n_components=10, n_neighbors=15)
    projection.fit_transform(make_blobs(n_samples=6, n_features=4, random_state=0)[0])
    params = projection.get_parameters()
    assert params["n_neighbors"] == 5
    assert params["n_components"] == 4


@pytest.mark.parametrize("mode", ["visualize", "transform"])
def test_transform_new_points_returns_component_count(X, mode):
    from velari_ai.integrations.sklearn.projection import ProjectionMode, UMAPProjection
    projection = UMAPProjection(mode=ProjectionMode(mode), n_components=3)
    projection.fit_transform(X)
    assert projection.transform(X[:5]).shape == (5, 3)


def test_transform_before_fit_raises_runtimeerror(X):
    from velari_ai.integrations.sklearn.projection import ProjectionMode, UMAPProjection
    with pytest.raises(RuntimeError):
        UMAPProjection(mode=ProjectionMode.TRANSFORM).transform(X)


def test_to_frame_with_labels_adds_label_column(X):
    from velari_ai.integrations.sklearn.projection import UMAPProjection
    df_proj = UMAPProjection().to_frame(X, labels=["a"] * 60)
    assert list(df_proj.columns) == ["umap_0", "umap_1", "label"]


def test_umap_fit_returns_self_and_enables_transform():
    from velari_ai.integrations.sklearn.projection import UMAPProjection
    X = make_blobs(n_samples=30, n_features=8, random_state=0)[0]
    projection = UMAPProjection(n_neighbors=5)
    assert projection.fit(X) is projection
    assert projection.transform(X[:3]).shape == (3, 2)
