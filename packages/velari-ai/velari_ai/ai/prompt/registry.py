import  pandas as pd
from    typing import Any, Optional
# package modules
from    velari_core.core.io.partition.hydra import read_hydra


class PromptRegistry(object):
    """Build and format prompts from a YAML-backed catalog of named templates.

    Args:
        uri (Optional[str]): Path to the YAML catalog file. When None, the catalog is empty.
        key (Optional[str]): Top-level key within the YAML file that contains the template
            list. When None, an empty dict is used as the catalog source.

    Examples:
        >>> registry = PromptRegistry(uri="config/prompts/catalog.yaml", key="prompts")
        >>> text = registry.format_template("billing_reminder", account_id="ACC-10293", due_date="2026-09-01")
    """
    def __init__(self, uri: Optional[str] = None, key: Optional[str] = None) -> None:
        self.df_catalog = pd.DataFrame(self.load_catalog(uri=uri, key=key))

    def load_catalog(self, uri: Optional[str] = None, key: Optional[str] = None) -> Any:
        """Load a named section from a Hydra YAML file and return it as a config object.

        Args:
            uri (Optional[str]): Path to the YAML file to read. When None, returns an empty dict.
            key (Optional[str]): Top-level key whose value is returned. When None, or when the
                key is absent from the file, an empty dict is returned.

        Returns:
            Any: The value stored under `key` in the parsed YAML — typically a list of
                `{name, template}` records — or an empty dict.
        """
        if uri is None or key is None:
            return {}
        cfg = read_hydra(filepath=uri)
        if cfg is None:
            return {}
        return cfg.get(key, {})

    def get_template(self, name: str) -> str:
        """Retrieve the raw template string for a named prompt entry.

        Args:
            name (str): Name of the prompt entry to look up in the catalog.

        Returns:
            str: Raw template string associated with the given name.

        Raises:
            KeyError: If no entry named `name` exists in the catalog.
        """
        try:
            matches = self.df_catalog[self.df_catalog["name"] == name]
            return matches.iloc[0]["template"]
        except (KeyError, IndexError) as e:
            raise KeyError(f"No prompt template registered under name {name!r}") from e

    def format_template(self, name: str, **kwargs: Any) -> str:
        """Retrieve and format a named template with the provided keyword arguments.

        Args:
            name (str): Name of the prompt entry to look up in the catalog.
            **kwargs (Any): Placeholder values to substitute into the template via `str.format`.

        Returns:
            str: Formatted string with all placeholders replaced by their corresponding values.

        Examples:
            >>> registry = PromptRegistry(uri="config/prompts/catalog.yaml", key="prompts")
            >>> text = registry.format_template("summarise", topic="climate change", length=200)
        """
        template = self.get_template(name)
        return template.format(**kwargs)
