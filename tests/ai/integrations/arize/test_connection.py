"""Tests for velari_ai.integrations.arize.connection."""

from typing import Any, Dict

import pytest
from omegaconf import OmegaConf


def _make_cfg(tmp_path, with_project=True):
    data: Dict[str, Any] = {
        "phoenix": {
            "connection": {"host": "localhost", "port": 6006},
            "storage": {"working_dir": str(tmp_path), "db_name": "phoenix.db"},
        }
    }
    if with_project:
        data["phoenix"]["project"] = {"project_name": "test-project"}
        data["phoenix"]["options"] = {"auto_instrument": True}
    return OmegaConf.create(data)


def _make_remote_cfg(with_project=True):
    data: Dict[str, Any] = {"phoenix": {"remote": {"endpoint": "https://phoenix.internal:6006"}}}
    if with_project:
        data["phoenix"]["project"] = {"project_name": "test-project"}
        data["phoenix"]["options"] = {"auto_instrument": True}
    return OmegaConf.create(data)


class TestConnectorConfig:
    def test_from_config_builds_connection_and_storage(self, tmp_path):
        from velari_ai.integrations.arize.connection import ConnectorConfig

        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=False))

        assert config.connection is not None
        assert config.storage is not None
        assert config.connection.host == "localhost"
        assert config.connection.port == 6006
        assert config.storage.db_name == "phoenix.db"
        assert config.project is None

    def test_from_config_builds_project_when_present(self, tmp_path):
        from velari_ai.integrations.arize.connection import ConnectorConfig

        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=True))

        assert config.project is not None
        assert config.project.project_name == "test-project"
        assert config.project.auto_instrument is True

    def test_storage_post_init_expands_and_creates_working_dir(self, tmp_path):
        from velari_ai.integrations.arize.connection import ConnectorConfig

        target = tmp_path / "phoenix_data"
        storage = ConnectorConfig.Storage(working_dir=str(target), db_name="phoenix.db")

        assert target.is_dir()
        assert storage.db_url == f"sqlite:///{target}/phoenix.db"

    def test_from_config_builds_remote_when_present(self):
        from velari_ai.integrations.arize.connection import ConnectorConfig

        config = ConnectorConfig.from_config(_make_remote_cfg(with_project=False))

        assert config.remote is not None
        assert config.remote.endpoint == "https://phoenix.internal:6006"
        assert config.connection is None
        assert config.storage is None


class TestConnector:
    def _patch_phoenix(self, monkeypatch):
        import velari_ai.integrations.arize.connection as connection

        class _FakeSession:
            def __init__(self, db_url, host, port):
                self.db_url = db_url
                self.host = host
                self.port = port
                self.url = f"http://{host}:{port}"

        captured = {}

        class _FakeClient:
            def __init__(self, base_url, api_key=None, headers=None):
                captured["client_base_url"] = base_url
                captured["client_api_key"]  = api_key
                captured["client_headers"]  = headers

        monkeypatch.setattr(connection, "ThreadSession", _FakeSession)
        monkeypatch.setattr(connection, "Client", _FakeClient)
        monkeypatch.setattr(connection, "register", lambda **kwargs: captured.update(register=kwargs))
        return captured

    def test_init_builds_session_and_client(self, tmp_path, monkeypatch):
        from velari_ai.integrations.arize.connection import Connector, ConnectorConfig

        captured = self._patch_phoenix(monkeypatch)
        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=False))
        connector = Connector(config)

        assert connector.url == "http://localhost:6006"
        assert captured["client_base_url"] == connector.url
        assert connector.config is config

    def test_init_registers_when_project_configured(self, tmp_path, monkeypatch):
        from velari_ai.integrations.arize.connection import Connector, ConnectorConfig

        captured = self._patch_phoenix(monkeypatch)
        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=True))
        Connector(config)

        assert captured["register"]["project_name"] == "test-project"
        assert captured["register"]["auto_instrument"] is True

    def test_init_skips_register_without_project(self, tmp_path, monkeypatch):
        from velari_ai.integrations.arize.connection import Connector, ConnectorConfig

        captured = self._patch_phoenix(monkeypatch)
        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=False))
        Connector(config)

        assert "register" not in captured

    def test_from_config_delegates_to_connector_config(self, tmp_path, monkeypatch):
        from velari_ai.integrations.arize.connection import Connector

        self._patch_phoenix(monkeypatch)
        connector = Connector.from_config(_make_cfg(tmp_path, with_project=False))

        assert connector.config.connection is not None
        assert connector.config.connection.host == "localhost"

    def test_get_tracer_returns_tracer_for_name(self, tmp_path, monkeypatch):
        from opentelemetry.trace import Tracer
        from velari_ai.integrations.arize.connection import Connector, ConnectorConfig

        self._patch_phoenix(monkeypatch)
        config = ConnectorConfig.from_config(_make_cfg(tmp_path, with_project=False))
        connector = Connector(config)

        assert isinstance(connector.get_tracer("velari-ai.test"), Tracer)

    def test_init_uses_remote_client_and_skips_thread_session(self, monkeypatch):
        import velari_ai.integrations.arize.connection as connection
        from velari_ai.integrations.arize.connection import Connector, ConnectorConfig

        captured = self._patch_phoenix(monkeypatch)

        def _fail_if_constructed(*args, **kwargs):
            raise AssertionError("ThreadSession must not be constructed in remote mode")

        monkeypatch.setattr(connection, "ThreadSession", _fail_if_constructed)

        config = ConnectorConfig.from_config(_make_remote_cfg(with_project=True))
        connector = Connector(config)

        assert connector.url == "https://phoenix.internal:6006"
        assert captured["client_base_url"] == "https://phoenix.internal:6006"
        assert captured["register"]["endpoint"] == "https://phoenix.internal:6006/v1/traces"
        assert captured["register"]["project_name"] == "test-project"

    @pytest.mark.parametrize(
        ("endpoint", "expected"),
        [
            ("https://phoenix.internal:6006", "https://phoenix.internal:6006/v1/traces"),
            ("https://phoenix.internal:6006/v1/traces", "https://phoenix.internal:6006/v1/traces"),
        ],
    )
    def test_init_appends_otlp_traces_path_to_remote_endpoint(self, monkeypatch, endpoint, expected):
        from velari_ai.integrations.arize.connection import Connector, ConnectorConfig

        captured = self._patch_phoenix(monkeypatch)
        config = ConnectorConfig(
            remote  = ConnectorConfig.Remote(endpoint=endpoint),
            project = ConnectorConfig.Project(project_name="test-project"),
        )
        Connector(config)

        assert captured["register"]["endpoint"] == expected

    def test_init_raises_without_remote_or_local_config(self):
        from velari_ai.integrations.arize.connection import Connector, ConnectorConfig

        with pytest.raises(ValueError):
            Connector(ConnectorConfig())
