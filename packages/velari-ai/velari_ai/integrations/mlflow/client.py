from __future__ import annotations
import  logging
from    typing import Any, Dict, List, Optional, Union
from    mlflow import MlflowClient
from    mlflow.exceptions import MlflowException
from    mlflow.entities.model_registry.model_version import ModelVersion
from    mlflow.entities.model_registry.registered_model import RegisteredModel
# package modules
from .connection import Connector

logger = logging.getLogger(__name__)


class ArtifactRegistry(object):
    """Server-side MLflow Model Registry for logged GenAI apps/agents (`LoggedModel`): register, alias, tag, list.

    Args:
        connector (Connector): Connected MLflow tracking server; scoped to this registry's own client, no global state.
    """
    def __init__(self, connector: Connector) -> None:
        self._client = MlflowClient(tracking_uri=connector.url)

    def register(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> ModelVersion:
        """Register a logged app under a name, creating it if new or adding a version if it already exists.

        Args:
            model_uri (str): Logged model to register, e.g. `info.model_uri` from `pyfunc.log_model()`.
            name (str): Registered name.
            tags (Optional[Dict[str, str]]): Version-level tags.
            **kwargs (Any): Passed to `MlflowClient.create_model_version` (e.g. `description`).

        Returns:
            ModelVersion: The newly created version.

        Examples:
            >>> registry = ArtifactRegistry(Connector.from_config(cfg))
            >>> info = mlflow.pyfunc.log_model(name="agent", python_model=billing_agent)
            >>> v = registry.register(info.model_uri, "billing-support-agent")
            >>> v.version
            1
        """
        try:
            self._client.get_registered_model(name)
        except MlflowException:
            self._client.create_registered_model(name)
        model_version = self._client.create_model_version(name, model_uri, tags=tags, **kwargs)
        logger.info(f"Registered app {name!r} version {model_version.version}")
        return model_version

    def set_alias(self, name: str, alias: str, version: Union[str, int]) -> None:
        """Point an alias at a registered version, moving it if it already exists.

        Args:
            name (str): Registered name.
            alias (str): Alias to set, e.g. `"champion"`.
            version (Union[str, int]): Version the alias should resolve to.

        Examples:
            >>> registry = ArtifactRegistry(Connector.from_config(cfg))
            >>> registry.set_alias("doc-retrieval-chain", "champion", version=4)
        """
        self._client.set_registered_model_alias(name, alias, str(version))
        logger.info(f"Alias {alias!r} on app {name!r} -> version {version}")

    def delete_alias(self, name: str, alias: str) -> None:
        """Remove an alias from a registered app; the underlying version is kept.

        Args:
            name (str): Registered name.
            alias (str): Alias to remove.

        Examples:
            >>> registry = ArtifactRegistry(Connector.from_config(cfg))
            >>> registry.delete_alias("doc-retrieval-chain", "challenger")
        """
        self._client.delete_registered_model_alias(name, alias)
        logger.info(f"Deleted alias {alias!r} from app {name!r}")

    def set_tag(self, name: str, key: str, value: str) -> None:
        """Set a registered-app-level tag.

        Args:
            name (str): Registered name.
            key (str): Tag key.
            value (str): Tag value.

        Examples:
            >>> registry = ArtifactRegistry(Connector.from_config(cfg))
            >>> registry.set_tag("compliance-review-agent", "owner", "risk-engineering")
        """
        self._client.set_registered_model_tag(name, key, value)

    def delete_tag(self, name: str, key: str) -> None:
        """Remove a registered-app-level tag.

        Args:
            name (str): Registered name.
            key (str): Tag key.

        Examples:
            >>> registry = ArtifactRegistry(Connector.from_config(cfg))
            >>> registry.delete_tag("compliance-review-agent", "owner")
        """
        self._client.delete_registered_model_tag(name, key)

    def list(self, filter_string: Optional[str] = None, max_results: Optional[int] = None) -> List[RegisteredModel]:
        """List registered apps, optionally filtered.

        Args:
            filter_string (Optional[str]): MLflow filter, e.g. `"name LIKE 'billing%'"`.
            max_results (Optional[int]): Cap on results; MLflow's default applies when omitted.

        Returns:
            List[RegisteredModel]: Matching registered apps.

        Examples:
            >>> registry = ArtifactRegistry(Connector.from_config(cfg))
            >>> [m.name for m in registry.list("name LIKE 'billing%'")]
            ['billing-support-agent', 'billing-dispute-triage']
        """
        options: Dict[str, Any] = {"filter_string": filter_string, "max_results": max_results}
        return list(self._client.search_registered_models(**{k: v for k, v in options.items() if v is not None}))
