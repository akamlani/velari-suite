from __future__ import annotations
import  pandas as pd
import  logging
from    typing import Any, TypeVar, Generic, Optional, Union, List, Dict, Tuple, Type, Hashable
from    dataclasses import dataclass, field
from    omegaconf import DictConfig
# package modules
from    velari_core.core.io.utils import trsfrm_frame_camelcase_to_snakecase as to_snakecase
from    .info import InfoSchema

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)
T    = TypeVar("T")

@dataclass
class DatasetExample:
    input: str
    target: Optional[str] = field(default=None)

@dataclass
class DatasetSpecInfo:
    """Information about the dataset"""
    data: Any
    name: str                   = field(default="")
    description: str            = field(default="")
    version: str                = field(default="0.0.1")
    metadata: Optional[dict]    = field(default_factory=dict)

    def __post_init__(self):
        self.dtype = type(self.data)

@dataclass
class DatasetSpecTabularInfo:
    """Information about the tabular dataset"""
    info: DatasetSpecInfo
    # post-processing info attributes
    dtype: Type[pd.DataFrame] = field(init=False)
    shape: Tuple[int, int]    = field(init=False)
    indicies: List[str]       = field(init=False)
    columns: List[str]        = field(init=False)
    num_rows: int             = field(init=False)
    num_cols: int             = field(init=False)

    def __post_init__(self):
        if self.info.data is None:
            raise ValueError("DatasetSpecInfo.data is None. Cannot initialize DatasetSpecTabularInfo.")

        self.dtype    = type(self.info.data)
        self.shape    = self.info.data.shape
        self.indicies = self.info.data.index
        self.columns  = self.info.data.columns
        self.num_rows, self.num_cols = self.info.data.shape

@dataclass
class DatasetProfileTimeSeries:
    @dataclass
    class TemporalSpan:
        start: pd.Timestamp
        end: pd.Timestamp
        days: float
        weeks: float
        years: float

    timespans: Dict[str, DatasetProfileTimeSeries.TemporalSpan] = field(default_factory=dict)


class DatasetT(Generic[T_co]):
    "Base Class for all Datasets"
    def __init__(self, spec: DatasetSpecInfo, **kwargs):
        self.spec = spec

    def info(self) -> dict:
        raise NotImplementedError("Method 'info' must be implemented in the subclass")

    def validate(self, schema: dict, **kwargs) -> bool:
        raise NotImplementedError("Method 'validate' must be implemented in the subclass")

    def transform(self, data: Any, target: Any, **kwargs):
        raise NotImplementedError("Method 'transform' must be implemented in the subclass")

    @staticmethod
    def load(path: str):
        raise NotImplementedError("Method 'load' must be implemented in the subclass")

    def save(self, data: Any, filepath: str, cache_dir: bool = True) -> None:
        raise NotImplementedError("Method 'save' must be implemented in the subclass")

    def preprocess(self, data: Any) -> Optional[Any]:
        raise NotImplementedError("Method 'preprocess' must be implemented in the subclass")

    def sample(self, **kwargs) -> Any:
        raise NotImplementedError("Method 'sample' must be implemented in the subclass")

    def __getitem__(self, index: int) -> Any:
        try:
            if self.spec.data is None:
                raise ValueError("Dataset spec has no data to index")
            return self.spec.data[index]
        except IndexError as e:
            raise IndexError(f"Index {index} out of bounds for dataset") from e
        except Exception as e:
            raise NotImplementedError(f"Dataset does not support indexing with {type(index)}: {e}") from e

    def __len__(self) -> int:
        try:
            if self.spec.data is None:
                raise ValueError("Dataset spec has no data to measure length")
            return len(self.spec.data)
        except Exception as e:
            raise NotImplementedError(f"Dataset does not support length operation: {e}") from e


class DatasetTabular(DatasetT[pd.DataFrame]):
    "Base Class for Tabular Datasets"
    def __init__(self, spec: DatasetSpecTabularInfo, **kwargs):
        super().__init__(spec.info, **kwargs)
        self.spec = spec
        self._df  = spec.info.data
        # self._profile = self._build_profile()

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        df = pd.read_csv(path).pipe(to_snakecase)
        logger.info(f"Loaded {len(df)} records from {path}")
        return df

    def __getitem__(self, index: int) -> Dict[Hashable, Any]:
        try:
            if self._df is None:
                raise ValueError("Dataset has not been loaded. Call 'load' method first.")
            return self._df.iloc[index].to_dict()
        except IndexError as e:
            raise IndexError(f"Index {index} out of bounds for dataset") from e
        except Exception as e:
            raise NotImplementedError(f"Dataset does not support indexing with {type(index)}: {e}") from e

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("Dataset has not been loaded. Call 'load' method first.")
        return self._df

    def to_records(self) -> List[Dict[Hashable, Any]]:
        if self._df is None:
            raise ValueError("Dataset has not been loaded. Call 'load' method first.")
        return self._df.to_dict(orient="records")

    def filter_condition(
        self,
        conditions: Dict[str, Any],
        inverse: bool = False,
    ) -> pd.DataFrame:
        """Filter rows by one or more column conditions.

        Builds a boolean mask by ANDing each column == value pair.
        Set ``inverse=True`` to exclude matching rows instead.

        Args:
            conditions: Mapping of column name to expected value.
                        All conditions are ANDed together.
            inverse:    If True, return rows that do NOT match.

        Returns:
            pd.DataFrame: Filtered subset of the dataset.

        Examples:
            # single column — keep rows where category == "billing"
            ds.filter_condition({"category": "billing"})

            # inverse — exclude rows where category == "billing"
            ds.filter_condition({"category": "billing"}, inverse=True)

            # multi-column — keep rows matching both conditions
            ds.filter_condition({"category": "billing", "priority": "high"})
        """
        mask = pd.Series(True, index=self.df.index)
        for col, val in conditions.items():
            mask &= self.df[col] == val
        return self.df.loc[~mask] if inverse else self.df.loc[mask]

    def sample(
        self,
        n_per_class: Optional[int] = None,
        stratify:    Optional[str] = None,
        max_samples: Optional[int] = None,
        seed:        Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        if stratify is not None:
            n_classes = self.df[stratify].nunique()
            k = max_samples // n_classes if max_samples is not None else n_per_class
            if n_per_class is not None and max_samples is not None:
                k = min(n_per_class, max_samples // n_classes)
            sampled = self.df.groupby(stratify).sample(n=k, random_state=seed).reset_index(drop=True)
            if not isinstance(sampled, pd.DataFrame):
                raise TypeError("groupby(...).sample(...) returned a Series, expected a DataFrame")
            return sampled
        return self.df.sample(n=max_samples if max_samples is not None else n_per_class, random_state=seed).reset_index(
            drop=True
        )


class DatasetTimeseries(DatasetTabular):
    """Base Class for Time Series Datasets"""
    def __init__(self, spec: DatasetSpecTabularInfo, **kwargs):
        super().__init__(spec, **kwargs)
        self.spec = spec
        self._df  = spec.info.data

    def temporal_span(self, column: str, unit: Optional[str] = None) -> DatasetProfileTimeSeries.TemporalSpan:
        column_series = self.df[column]
        if not isinstance(column_series, pd.Series):
            raise TypeError(f"expected column {column!r} to be a Series, got {type(column_series).__name__}")
        series = InfoSchema._cast_datetime(column_series, unit)
        start, end = series.min(), series.max()
        delta = end - start
        return DatasetProfileTimeSeries.TemporalSpan(
            start=start,
            end=end,
            days=float(delta.days),
            weeks=round(delta.days / 7, 2),
            years=round(delta.days / 365.25, 2),
        )
