"""Tests for velari_core.integrations.fastapi.server."""

_CONFIG_YAML = """
info:
  title:       Test Service
  description: A test service
  version:     0.0.1
  docs_url:    /docs
  redoc_url:   /redoc
connection:
  host:        127.0.0.1
  port:        9000
cors:
  allow_origins:     ["*"]
  allow_credentials: true
  allow_methods:     ["*"]
  allow_headers:     ["*"]
  expose_headers:    []
"""


def _write_config(tmp_path):
    path = tmp_path / "service.yaml"
    path.write_text(_CONFIG_YAML)
    return str(path)


def test_build_sets_app_title_from_config_info_title(tmp_path):
    from velari_core.integrations.fastapi.server import Server

    server = Server.from_yaml(_write_config(tmp_path))

    assert server.app.title == "Test Service"


def test_build_does_not_leak_host_port_into_app_extra(tmp_path):
    from velari_core.integrations.fastapi.server import Server

    server = Server.from_yaml(_write_config(tmp_path))

    assert "host" not in server.app.extra
    assert "port" not in server.app.extra


def test_run_still_binds_host_and_port_from_config(tmp_path, monkeypatch):
    from velari_core.integrations.fastapi.server import Server
    import uvicorn

    server = Server.from_yaml(_write_config(tmp_path))
    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))

    server.run()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9000
