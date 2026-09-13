from __future__ import annotations

import logging
import os

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional
import mlflow
from mlflow.client import MlflowClient
from omegaconf import DictConfig
# package modules
from ...ai.tracing.trace import SpanLike, TracerLike, TracingConnector

logger = logging.getLogger(__name__)


class _MlflowTracer:
    def __init__(self, name: str, version: Optional[str] = None) -> None:
        self._name    = name
        self._version = version

    @contextmanager
    def start_as_current_span(self, name: str) -> Iterator[SpanLike]:
        with mlflow.start_span(name=name) as span:
            span.set_attribute("tracer.name", self._name)
            if self._version:
                span.set_attribute("tracer.version", self._version)
            yield span


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
        if "project" in cfg.mlflow or "options" in cfg.mlflow:
            project = cls.Project(
                project_name    = cfg.mlflow.project.project_name if "project" in cfg.mlflow else None,
                auto_instrument = cfg.mlflow.options.auto_instrument if "options" in cfg.mlflow else False,
            )

        if "remote" in cfg.mlflow:
            remote = cls.Remote(
                endpoint = cfg.mlflow.remote.endpoint,
                api_key  = cfg.mlflow.remote.api_key if "api_key" in cfg.mlflow.remote else None,
                headers  = dict(cfg.mlflow.remote.headers) if "headers" in cfg.mlflow.remote else None,
            )
            return cls(remote=remote, project=project)

        connection = cls.Connection(
            host = cfg.mlflow.connection.host,
            port = cfg.mlflow.connection.port,
        )
        storage = cls.Storage(
            working_dir = cfg.mlflow.storage.working_dir,
            db_name     = cfg.mlflow.storage.db_name,
        )
        return cls(connection=connection, storage=storage, project=project)


class Connector(TracingConnector):
    def __init__(self, config: ConnectorConfig) -> None:
        self._config = config
        if config.remote is not None:
            self._url = config.remote.endpoint
            mlflow.set_tracking_uri(config.remote.endpoint)
            if config.remote.api_key:
                os.environ["MLFLOW_TRACKING_TOKEN"] = config.remote.api_key
            logger.info(f"Connected to remote MLflow — {self._url}")
        else:
            if config.connection is None or config.storage is None:
                raise ValueError("ConnectorConfig requires either 'remote', or both 'connection' and 'storage'")
            self._url = config.storage.db_url
            mlflow.set_tracking_uri(config.storage.db_url)
            logger.info(f"MLflow tracking to {self._url}")

        self._client = MlflowClient(tracking_uri=self._url)
        if config.project and config.project.project_name:
            mlflow.set_experiment(config.project.project_name)
        if config.project and config.project.auto_instrument:
            mlflow.autolog()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Connector:
        return cls(ConnectorConfig.from_config(cfg))

    def get_tracer(self, name: str, version: Optional[str] = None) -> TracerLike:
        return _MlflowTracer(name, version)

    @property
    def config(self) -> ConnectorConfig:
        return self._config

    @property
    def url(self) -> str:
        return self._url

    @property
    def client(self) -> MlflowClient:
        return self._client
