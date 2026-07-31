from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Generic, List, Mapping, Optional, Callable, Protocol, Type, TypeVar, runtime_checkable
from uuid import uuid4
from pydantic import BaseModel


@runtime_checkable
class DocModel(Protocol):
    """Structural type for document inputs: dict/TypedDict, Pydantic BaseModel, or dataclass."""


V = TypeVar("V")


class KeyValueStore(ABC, Generic[V]):
    @abstractmethod
    def put(self, key: str, value: V) -> None: ...

    @abstractmethod
    def get(self, key: str) -> Optional[V]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def has(self, key: str) -> bool: ...

    @abstractmethod
    def keys(self) -> List[str]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def __getitem__(self, key: str) -> V:
        if not self.has(key):
            raise KeyError(key)
        return self.get(key)  # type: ignore[return-value]

    def __setitem__(self, key: str, value: V) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        if not self.has(key):
            raise KeyError(key)
        self.delete(key)

    def __contains__(self, key: object) -> bool:
        return self.has(key) if isinstance(key, str) else False


class LocalKeyValueStore(KeyValueStore[V]):
    def __init__(self, data: Optional[Dict[str, V]] = None) -> None:
        self._store: Dict[str, V] = dict(data) if data else {}

    def __len__(self) -> int:
        return len(self._store)

    def filter(self, predicate: Callable[[str, V], bool]) -> Dict[str, V]:
        return {k: v for k, v in self._store.items() if predicate(k, v)}

    def put(self, key: str, value: V) -> None:
        self._store[key] = value

    def get(self, key: str) -> Optional[V]:
        return self._store.get(key)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def has(self, key: str) -> bool:
        return key in self._store

    def keys(self) -> List[str]:
        return list(self._store.keys())


class DocumentStore(ABC):
    @abstractmethod
    def upsert(self, documents: List[DocModel]) -> None: ...

    @abstractmethod
    def get(
        self,
        ids: List[str],
        schema: Optional[Type] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None: ...

    @abstractmethod
    def has(self, id: str) -> bool: ...

    @abstractmethod
    def ids(self) -> List[str]: ...

    @abstractmethod
    def to_frame(self) -> pd.DataFrame: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def __contains__(self, id: object) -> bool:
        return self.has(id) if isinstance(id, str) else False


class LocalDocumentStore(DocumentStore):
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._store)

    def has(self, id: str) -> bool:
        return id in self._store

    def ids(self) -> List[str]:
        return list(self._store.keys())

    def delete(self, ids: List[str]) -> None:
        for id_ in ids:
            self._store.pop(id_, None)

    @staticmethod
    def _to_dict(doc: DocModel) -> Dict[str, Any]:
        if isinstance(doc, BaseModel):
            return doc.model_dump()
        if is_dataclass(doc) and not isinstance(doc, type):
            return asdict(doc)
        if isinstance(doc, Mapping):
            return dict(doc)
        raise TypeError(f"Unsupported document type: {type(doc)!r}")

    def upsert(self, documents: List[DocModel]) -> None:
        for doc in documents:
            d = self._to_dict(doc)
            doc_id = str(d.get("id", uuid4()))
            self._store[doc_id] = d

    def get(
        self,
        ids: List[str],
        schema: Optional[Type] = None,
    ) -> List[Dict[str, Any]]:
        docs = [self._store[id_] for id_ in ids if id_ in self._store]
        if schema is None:
            return docs
        return [schema(**doc) for doc in docs]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._store.values())
