from __future__ import annotations

import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
from omegaconf import DictConfig
from opentelemetry import trace
from phoenix.client import Client
from phoenix.otel import register
from phoenix.session.session import ThreadSession

logger = logging.getLogger(__name__)


def _otlp_http_endpoint(base_url: str) -> str:
    path = base_url.rstrip("/")
    return path if path.endswith("/v1/traces") else f"{path}/v1/traces"


@dataclass
class ConnectorConfig:
    @dataclass
    class Connection:
        host: str
        port: int

    @dataclass
    class Storage:
        working_dir: str
        db_name:     str

        def __post_init__(self) -> None:
            self.working_dir = str(Path(self.working_dir).expanduser())
            Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        @property
        def db_url(self) -> str:
            return f"sqlite:///{self.working_dir}/{self.db_name}"

    @dataclass
    class Project:
        project_name:    Optional[str] = field(default=None)
        auto_instrument: bool          = field(default=False)

    @dataclass
    class Remote:
        endpoint: str
        api_key:  Optional[str]            = field(default=None)
        headers:  Optional[Dict[str, str]] = field(default=None)

    connection: Optional[Connection] = field(default=None)
    storage:    Optional[Storage]    = field(default=None)
    project:    Optional[Project]    = field(default=None)
    remote:     Optional[Remote]     = field(default=None)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> ConnectorConfig:
        project = None
        if "project" in cfg.phoenix or "options" in cfg.phoenix:
            project = cls.Project(
                project_name    = cfg.phoenix.project.project_name if "project" in cfg.phoenix else None,
                auto_instrument = cfg.phoenix.options.auto_instrument if "options" in cfg.phoenix else False,
            )

        if "remote" in cfg.phoenix:
            remote = cls.Remote(
                endpoint = cfg.phoenix.remote.endpoint,
                api_key  = cfg.phoenix.remote.api_key if "api_key" in cfg.phoenix.remote else None,
                headers  = dict(cfg.phoenix.remote.headers) if "headers" in cfg.phoenix.remote else None,
            )
            return cls(remote=remote, project=project)

        connection = cls.Connection(
            host = cfg.phoenix.connection.host,
            port = cfg.phoenix.connection.port,
        )
        storage = cls.Storage(
            working_dir = cfg.phoenix.storage.working_dir,
            db_name     = cfg.phoenix.storage.db_name,
        )
        return cls(connection=connection, storage=storage, project=project)


class Connector(object):
    def __init__(self, config: ConnectorConfig) -> None:
        self._config = config
        if config.remote is not None:
            self._session = None
            self._url     = config.remote.endpoint
            self._client  = Client(base_url=config.remote.endpoint, api_key=config.remote.api_key, headers=config.remote.headers)
            if config.project and config.project.project_name:
                register(
                    endpoint        = _otlp_http_endpoint(config.remote.endpoint),
                    project_name    = config.project.project_name,
                    auto_instrument = config.project.auto_instrument,
                    verbose         = False,
                    protocol        = "http/protobuf",
                    headers         = config.remote.headers,
                )
            logger.info(f"Connected to remote Phoenix — {self._url}")
        else:
            if config.connection is None or config.storage is None:
                raise ValueError("ConnectorConfig requires either 'remote', or both 'connection' and 'storage'")
            self._session = ThreadSession(config.storage.db_url, host=config.connection.host, port=config.connection.port)
            self._url     = self._session.url
            self._client  = Client(base_url=self._session.url)
            if config.project and config.project.project_name:
                register(
                    project_name    = config.project.project_name,
                    auto_instrument = config.project.auto_instrument,
                    verbose         = False,
                    protocol        = "http/protobuf",
                )
            logger.info(f"Phoenix started — {self._url}")

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Connector:
        return cls(ConnectorConfig.from_config(cfg))

    def get_tracer(self, name: str, version: Optional[str] = None):
        return trace.get_tracer(name, version)

    @property
    def config(self) -> ConnectorConfig:
        return self._config

    @property
    def url(self) -> str:
        return self._url

    @property
    def client(self) -> Client:
        return self._client
