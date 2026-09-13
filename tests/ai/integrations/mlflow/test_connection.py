"""Tests for velari_ai.integrations.mlflow.connection."""

import os

from contextlib import contextmanager
from typing import Any, Dict

import pytest
from omegaconf import OmegaConf

pytest.importorskip("mlflow")


def _make_cfg(tmp_path, with_project=True):
    data: Dict[str, Any] = {
        "mlflow": {
            "connection": {"host": "localhost", "port": 5000},
            "storage": {"working_dir": str(tmp_path), "db_name": "mlflow.db"},
        }
    }
    if with_project:
        data["mlflow"]["project"] = {"project_name": "test-project"}
        data["mlflow"]["options"] = {"auto_instrument": True}
    return OmegaConf.create(data)


def _make_remote_cfg(with_project=True):
    data: Dict[str, Any] = {"mlflow": {"remote": {"endpoint": "https://mlflow.internal:5000"}}}
    if with_project:
        data["mlflow"]["project"] = {"project_name": "test-project"}
        data["mlflow"]["options"] = {"auto_instrument": True}
    return OmegaConf.create(data)


class TestConnectorConfig:
    def test_from_config_builds_connection_and_storage(self, tmp_path):
        from velari_ai.integrations.mlflow.connection import ConnectorConfig

        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=False))

        assert config.connection is not None
        assert config.storage is not None
        assert config.connection.host == "localhost"
        assert config.connection.port == 5000
        assert config.storage.db_name == "mlflow.db"
        assert config.project is None

    def test_from_config_builds_project_when_present(self, tmp_path):
        from velari_ai.integrations.mlflow.connection import ConnectorConfig

        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=True))

        assert config.project is not None
        assert config.project.project_name == "test-project"
        assert config.project.auto_instrument is True

    def test_storage_post_init_expands_and_creates_working_dir(self, tmp_path):
        from velari_ai.integrations.mlflow.connection import ConnectorConfig

        target = tmp_path / "mlflow_data"
        storage = ConnectorConfig.Storage(working_dir=str(target), db_name="mlflow.db")

        assert target.is_dir()
        assert storage.db_url == f"sqlite:///{target}/mlflow.db"

    def test_from_config_builds_remote_when_present(self):
        from velari_ai.integrations.mlflow.connection import ConnectorConfig

        config = ConnectorConfig.from_config(_make_remote_cfg(with_project=False))

        assert config.remote is not None
        assert config.remote.endpoint == "https://mlflow.internal:5000"
        assert config.connection is None
        assert config.storage is None


class TestConnector:
    def _patch_mlflow(self, monkeypatch):
        import velari_ai.integrations.mlflow.connection as connection

        captured = {}

        class _FakeClient:
            def __init__(self, tracking_uri=None):
                captured["client_tracking_uri"] = tracking_uri

        monkeypatch.setattr(connection.mlflow, "set_tracking_uri", lambda uri: captured.update(tracking_uri=uri))
        monkeypatch.setattr(connection.mlflow, "set_experiment", lambda name: captured.update(set_experiment=name))
        monkeypatch.setattr(connection.mlflow, "autolog", lambda: captured.update(autolog=True))
        monkeypatch.setattr(connection, "MlflowClient", _FakeClient)
        return captured

    def test_init_sets_tracking_uri_and_client(self, tmp_path, monkeypatch):
        from velari_ai.integrations.mlflow.connection import Connector, ConnectorConfig

        captured = self._patch_mlflow(monkeypatch)
        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=False))
        connector = Connector(config)

        assert connector.url == f"sqlite:///{tmp_path}/mlflow.db"
        assert captured["tracking_uri"] == connector.url
        assert captured["client_tracking_uri"] == connector.url
        assert connector.config is config

    def test_init_configures_experiment_and_autolog_when_project_configured(self, tmp_path, monkeypatch):
        from velari_ai.integrations.mlflow.connection import Connector, ConnectorConfig

        captured = self._patch_mlflow(monkeypatch)
        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=True))
        Connector(config)

        assert captured["set_experiment"] == "test-project"
        assert captured["autolog"] is True

    def test_init_skips_experiment_and_autolog_without_project(self, tmp_path, monkeypatch):
        from velari_ai.integrations.mlflow.connection import Connector, ConnectorConfig

        captured = self._patch_mlflow(monkeypatch)
        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=False))
        Connector(config)

        assert "set_experiment" not in captured
        assert "autolog" not in captured

    def test_from_config_delegates_to_connector_config(self, tmp_path, monkeypatch):
        from velari_ai.integrations.mlflow.connection import Connector

        self._patch_mlflow(monkeypatch)
        connector = Connector.from_config(_make_cfg(tmp_path, with_project=False))

        assert connector.config.connection is not None
        assert connector.config.connection.host == "localhost"

    def test_init_uses_remote_tracking_uri(self, monkeypatch):
        from velari_ai.integrations.mlflow.connection import Connector, ConnectorConfig

        captured = self._patch_mlflow(monkeypatch)
        config = ConnectorConfig.from_config(_make_remote_cfg(with_project=True))
        connector = Connector(config)

        assert connector.url == "https://mlflow.internal:5000"
        assert captured["tracking_uri"] == "https://mlflow.internal:5000"
        assert captured["client_tracking_uri"] == "https://mlflow.internal:5000"
        assert captured["set_experiment"] == "test-project"

    def test_init_sets_tracking_token_env_var_from_remote_api_key(self, monkeypatch):
        from velari_ai.integrations.mlflow.connection import Connector, ConnectorConfig

        self._patch_mlflow(monkeypatch)
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        config = ConnectorConfig(remote=ConnectorConfig.Remote(endpoint="https://mlflow.internal:5000", api_key="secret-token"))
        Connector(config)

        assert os.environ["MLFLOW_TRACKING_TOKEN"] == "secret-token"

    def test_init_skips_tracking_token_env_var_without_api_key(self, monkeypatch):
        from velari_ai.integrations.mlflow.connection import Connector, ConnectorConfig

        self._patch_mlflow(monkeypatch)
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        config = ConnectorConfig(remote=ConnectorConfig.Remote(endpoint="https://mlflow.internal:5000"))
        Connector(config)

        assert "MLFLOW_TRACKING_TOKEN" not in os.environ

    def test_get_tracer_start_as_current_span_sets_name_and_version_attributes(self, tmp_path, monkeypatch):
        import velari_ai.integrations.mlflow.connection as connection
        from velari_ai.integrations.mlflow.connection import Connector, ConnectorConfig

        self._patch_mlflow(monkeypatch)
        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=False))
        connector = Connector(config)

        class _FakeSpan:
            def __init__(self):
                self.attributes = {}

            def set_attribute(self, key, value):
                self.attributes[key] = value

        fake_span = _FakeSpan()

        @contextmanager
        def _fake_start_span(name):
            yield fake_span

        monkeypatch.setattr(connection.mlflow, "start_span", _fake_start_span)

        tracer = connector.get_tracer("velari-ai.test", "1.0")
        with tracer.start_as_current_span("example-span") as span:
            span.set_attribute("example.attribute", "hello-mlflow")

        assert fake_span.attributes["tracer.name"] == "velari-ai.test"
        assert fake_span.attributes["tracer.version"] == "1.0"
        assert fake_span.attributes["example.attribute"] == "hello-mlflow"

    def test_init_raises_without_remote_or_local_config(self):
        from velari_ai.integrations.mlflow.connection import Connector, ConnectorConfig

        with pytest.raises(ValueError):
            Connector(ConnectorConfig())
