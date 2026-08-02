import pandas as pd
from   typing import Optional

class InfoSchema(object):
    @staticmethod
    def _cast_datetime(series: pd.Series, unit: Optional[str] = None) -> pd.Series:
        if unit == "unix_s":
            return pd.to_datetime(series, unit="s", errors="coerce")
        if unit == "unix_ms":
            return pd.to_datetime(series, unit="ms", errors="coerce")
        return pd.to_datetime(series, errors="coerce")
