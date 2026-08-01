import pandas as pd
import importlib
import importlib.metadata


def get_package_versions() -> pd.DataFrame:
    data = {
        distribution.metadata["Name"].lower(): distribution.version for distribution in importlib.metadata.distributions()
    }
    return pd.DataFrame.from_dict(data, orient="index", columns=["version"]).sort_index()
