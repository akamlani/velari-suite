from __future__ import annotations
import  logging
import  mlflow
from    typing import Any, Dict, List, Optional, Union
from    mlflow import genai
from    mlflow.entities.model_registry import Prompt, PromptVersion
# package modules
from .connection import Connector

logger = logging.getLogger(__name__)


class PromptRegistry(object):
    """MLflow Prompt Registry wrapper: register, load, alias, tag, and list versioned prompts.

    Args:
        connector (Connector): Connected MLflow tracking server; its URI is (re)applied so `mlflow.genai` calls target it.
    """
    def __init__(self, connector: Connector) -> None:
        mlflow.set_tracking_uri(connector.url)

    def register(
        self,
        name: str,
        template: Union[str, List[Dict[str, Any]]],
        commit_message: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> PromptVersion:
        """Register a prompt, creating it if new or adding a version if it already exists.

        Args:
            name (str): Prompt name.
            template (Union[str, List[Dict[str, Any]]]): Text with `{{variables}}`, or chat messages.
            commit_message (Optional[str]): Description of this version's change.
            tags (Optional[Dict[str, str]]): Version-level tags.
            **kwargs (Any): Passed to MLflow as-is (e.g. `response_format`, `model_config`).

        Returns:
            PromptVersion: The newly created version.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> v = registry.register(
            ...     "ticket-triage",
            ...     "Classify the severity of this support ticket: {{ticket_text}}",
            ...     commit_message="Add severity rubric",
            ... )
            >>> v.version
            1
        """
        options = {"commit_message": commit_message, "tags": tags}
        prompt = genai.register_prompt(name, template, **{k: v for k, v in options.items() if v is not None}, **kwargs)
        logger.info(f"Registered prompt {name!r} version {prompt.version}")
        return prompt

    def load(self, name: str, version: Optional[Union[str, int]] = None, alias: Optional[str] = None) -> PromptVersion:
        """Load a prompt by version, by alias, or (with neither) its latest version.

        Args:
            name (str): Prompt name.
            version (Optional[Union[str, int]]): Specific version; mutually exclusive with `alias`.
            alias (Optional[str]): Alias to resolve, e.g. `"production"`.

        Returns:
            PromptVersion: The resolved prompt version.

        Raises:
            ValueError: If both `version` and `alias` are given.
            MlflowException: If the prompt, version, or alias doesn't exist.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> prompt = registry.load("churn-report-summary", alias="production")
            >>> prompt.format(quarter="Q3", segment="enterprise")
        """
        if version is not None and alias is not None:
            raise ValueError("Pass either 'version' or 'alias', not both")
        return genai.load_prompt(f"prompts:/{name}@{alias}" if alias else name, version=version)

    def set_alias(self, name: str, alias: str, version: int) -> None:
        """Point an alias at a prompt version, moving it if it already exists.

        Args:
            name (str): Prompt name.
            alias (str): Alias to set, e.g. `"staging"`.
            version (int): Version the alias should resolve to.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> registry.set_alias("contract-clause-extractor", "production", version=3)
        """
        genai.set_prompt_alias(name, alias, version)
        logger.info(f"Alias {alias!r} on prompt {name!r} -> version {version}")

    def delete_alias(self, name: str, alias: str) -> None:
        """Remove an alias from a prompt; the underlying version is kept.

        Args:
            name (str): Prompt name.
            alias (str): Alias to remove.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> registry.delete_alias("contract-clause-extractor", "staging")
        """
        genai.delete_prompt_alias(name, alias)
        logger.info(f"Deleted alias {alias!r} from prompt {name!r}")

    def set_tag(self, name: str, key: str, value: str) -> None:
        """Set a prompt-level tag.

        Args:
            name (str): Prompt name.
            key (str): Tag key.
            value (str): Tag value.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> registry.set_tag("kb-answer-grounding", "owner", "retrieval-team")
        """
        genai.set_prompt_tag(name, key, value)

    def delete_tag(self, name: str, key: str) -> None:
        """Remove a prompt-level tag.

        Args:
            name (str): Prompt name.
            key (str): Tag key.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> registry.delete_tag("kb-answer-grounding", "owner")
        """
        genai.delete_prompt_tag(name, key)

    def list(self, filter_string: Optional[str] = None, max_results: Optional[int] = None) -> List[Prompt]:
        """List registered prompts, optionally filtered.

        Args:
            filter_string (Optional[str]): MLflow filter, e.g. `"name LIKE 'ticket%'"`.
            max_results (Optional[int]): Cap on results; MLflow's default applies when omitted.

        Returns:
            List[Prompt]: Matching prompts.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> [p.name for p in registry.list("name LIKE 'ticket%'")]
            ['ticket-triage', 'ticket-summary']
        """
        options = {"filter_string": filter_string, "max_results": max_results}
        return list(genai.search_prompts(**{k: v for k, v in options.items() if v is not None}))
