from __future__ import annotations

import pandas as pd
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional, Sequence
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


class RecordRegistry(Registry):
    """Named records held in a DataFrame: one row per record, one column per attribute.

    Args:
        df_records (Optional[pd.DataFrame]): The records; `None` starts empty.
        key (str): The column holding each record's unique name.

    Examples:
        >>> registry = RecordRegistry(pd.DataFrame({"name": ["exact_match", "llm_judge"], "kind": ["heuristic", "model"]}))
        >>> registry.list(kind="model")
        ['llm_judge']
    """
    def __init__(self, df_records: Optional[pd.DataFrame] = None, key: str = "name") -> None:
        self._key = key
        self.create(df_records if df_records is not None else pd.DataFrame({key: []}))

    def create(self, df_records: pd.DataFrame) -> pd.DataFrame:
        """Replace the records.

        Args:
            df_records (pd.DataFrame): One row per record.

        Returns:
            pd.DataFrame: A copy of the stored records.

        Raises:
            ValueError: If the key column is missing, or a name is empty, not a string, or repeated.
        """
        names = df_records.get(self._key)
        if names is None or not all(isinstance(name, str) and name for name in names) or names.duplicated().any():
            raise ValueError(f"records need a column {self._key!r} of unique, non-empty string names")
        self._frame = df_records.reset_index(drop=True)
        return self.to_dataframe()

    def get(self, name: str) -> Any:
        """Get a record: a dict of its columns, with missing values as `None`; a subclass may return a richer object.

        Args:
            name (str): The record's name.

        Returns:
            Any: Column to value, by default.

        Raises:
            KeyError: If no record is named `name`.
        """
        rows = self._frame[self._frame[self._key] == name]
        if rows.empty:
            raise KeyError(name)
        record = rows.iloc[0].astype(object)
        return record.where(record.notna(), None).to_dict()

    def list(self, **filters: Any) -> List[str]:
        """List the names in order, keeping only records whose columns equal every filter.

        Args:
            **filters (Any): Column to the value a record must have, such as `kind="model"`.

        Returns:
            List[str]: The matching names.

        Raises:
            KeyError: If a filter names a column the records do not have.
        """
        matches = (self._frame[[*filters]] == pd.Series(filters, dtype=object)).fillna(False).all(axis=1)
        return self._frame.loc[matches, self._key].tolist()

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame.copy()

    def to_xml(self, columns: Sequence[str], root: str = "records", row: str = "record", names: Optional[Sequence[str]] = None) -> str:
        """Render the chosen columns of the records as XML: one `row` element per record inside `root`.

        A missing value, or a column the records lack, leaves that element out. A dotted column such as `meta.category`
        becomes the tag `category`.

        Args:
            columns (Sequence[str]): The columns to render, in this order.
            root (str): The outer tag.
            row (str): The tag of each record.
            names (Optional[Sequence[str]]): Only these records, in table order; `None` renders every record.

        Returns:
            str: The XML text, indented; an empty string when there are no records.

        Raises:
            KeyError: If a name in `names` is not a record.

        Examples:
            >>> registry = RecordRegistry(pd.DataFrame({"name": ["exact_match", "llm_judge"], "kind": ["heuristic", None]}))
            >>> print(registry.to_xml(["name", "kind"], root="scorers", row="scorer", names=["llm_judge"]))
            <scorers>
              <scorer>
                <name>llm_judge</name>
              </scorer>
            </scorers>
        """
        df_records = self.to_dataframe()
        if names is not None:
            if unknown := set(names) - set(self.list()):
                raise KeyError(sorted(unknown))
            df_records = df_records.loc[df_records[self._key].isin(names)]
        tree = ET.Element(root)
        for _, record in df_records.reindex(columns=[*columns]).iterrows():
            element = ET.SubElement(tree, row)
            for column, value in record.items():
                *_, tag = str(column).split(".")
                text = None
                if isinstance(value, (list, tuple)):
                    text = " ".join(map(str, value)) or None   # a list cell is one text; an empty list is missing
                elif pd.notna(value):
                    text = str(value)
                if text is not None:
                    ET.SubElement(element, tag).text = text
        ET.indent(tree)
        return ET.tostring(tree, encoding="unicode") if len(tree) else ""


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
