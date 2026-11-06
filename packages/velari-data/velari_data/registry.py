from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional
# package modules
from .storage import LocalKeyValueStore


class Registry(ABC):
    @abstractmethod
    def get(self, name: str, /) -> Any: ...

    @abstractmethod
    def list(self, **kwargs: Any) -> List[Any]: ...

    def create(self, *args: Any, **kwargs: Any) -> Any:
        cfg = args[0]
        self._catalog = LocalKeyValueStore({str(k): v for k, v in dict(cfg).items()})
        return self.to_dataframe()

    def has(self, name: str) -> bool:
        try:
            self.get(name)
            return True
        except Exception:
            return False

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __len__(self) -> int:
        return len(self.list())

    def __iter__(self) -> Iterator[Any]:
        return iter(self.list())

    def to_dataframe(self) -> pd.DataFrame:
        names = self._catalog.keys()
        return pd.DataFrame({"name": names, "value": [self._catalog.get(name) for name in names]})


class DatasetRegistryInterface(Registry):
    """Shared contract for backend dataset registries: `create`, `extend`, plus `Registry`'s `get`/`list`.

    Backends may accept more than declared here (extra optional parameters), never less.
    """
    @abstractmethod
    def create(
        self,
        name: str,
        df_records: pd.DataFrame,
        input_keys: List[str],
        output_keys: Optional[List[str]] = None,
        description: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a dataset from DataFrame rows.

        Args:
            name (str): Dataset name.
            df_records (pd.DataFrame): Rows to upload.
            input_keys (List[str]): Columns holding record inputs.
            output_keys (Optional[List[str]]): Columns holding expected outputs.
            description (Optional[str]): Human-readable dataset description.
            metadata_keys (Optional[List[str]]): Columns holding per-record metadata.
            **kwargs (Any): Backend-specific options.

        Returns:
            Any: The backend's dataset object.
        """

    @abstractmethod
    def extend(
        self,
        dataset: Any,
        df_records: pd.DataFrame,
        input_keys: List[str],
        output_keys: Optional[List[str]] = None,
        metadata_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Add DataFrame rows to an existing dataset; how overlapping rows are handled is backend-defined.

        Args:
            dataset (Any): The dataset to extend: a name or the backend's dataset object.
            df_records (pd.DataFrame): Rows to add.
            input_keys (List[str]): Columns holding record inputs.
            output_keys (Optional[List[str]]): Columns holding expected outputs.
            metadata_keys (Optional[List[str]]): Columns holding per-record metadata.
            **kwargs (Any): Backend-specific options.

        Returns:
            Any: The backend's updated dataset object.
        """
