from    typing import Any, List, Optional
import  pandas as pd
from    omegaconf import OmegaConf
# package modules
from    velari_core.core.io.partition.hydra import read_hydra
from    velari_data.registry import Registry


class PromptRegistry(Registry):
    """Build and format prompts from a YAML-backed catalog of named templates.

    Args:
        uri (Optional[str]): Path to the YAML catalog file — a flat mapping of prompt name to
            template string. When None, the catalog is empty.

    Examples:
        >>> registry = PromptRegistry(uri="config/prompts/catalog.yaml")
        >>> text = registry.format_template("billing_reminder", account_id="ACC-10293", due_date="2026-09-01")
    """
    def __init__(self, uri: Optional[str] = None) -> None:
        self.create(uri=uri)

    def create(self, uri: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load a flat name→template mapping from a YAML file into the catalog.

        Args:
            uri (Optional[str]): Path to the YAML file to read. When None, the catalog is empty.

        Returns:
            pd.DataFrame: The loaded catalog as a name/value DataFrame (via `to_dataframe()`).
        """
        cfg = read_hydra(filepath=uri) if uri is not None else None
        return super().create(cfg if cfg is not None else OmegaConf.create({}))

    def get(self, name: str, **kwargs) -> str:
        """Retrieve the raw template string for a named prompt entry.

        Args:
            name (str): Name of the prompt entry to look up in the catalog.

        Returns:
            str: Raw template string associated with the given name.

        Raises:
            KeyError: If no entry named `name` exists in the catalog.
        """
        return self._catalog[name]

    def list(self, **kwargs: Any) -> List[str]:
        """List the names of all prompt entries in the catalog.

        Returns:
            List[str]: Registered entry names, or an empty list if the catalog is empty.

        Examples:
            >>> registry = PromptRegistry(uri="config/prompts/catalog.yaml")
            >>> registry.list()
            ['summarise', 'classify', 'billing_reminder']
        """
        return self._catalog.keys()

    def format_template(self, name: str, **kwargs: Any) -> str:
        """Retrieve and format a named template with the provided keyword arguments.

        Args:
            name (str): Name of the prompt entry to look up in the catalog.
            **kwargs (Any): Placeholder values to substitute into the template via `str.format`.

        Returns:
            str: Formatted string with all placeholders replaced by their corresponding values.

        Examples:
            >>> registry = PromptRegistry(uri="config/prompts/catalog.yaml")
            >>> text = registry.format_template("summarise", topic="climate change", length=200)
        """
        template = self.get(name)
        return template.format(**kwargs)
