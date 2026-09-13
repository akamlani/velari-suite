from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Iterator, List
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
