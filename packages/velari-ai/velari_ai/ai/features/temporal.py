import numpy  as np
import pandas as pd

# ADI/CV2 is to route SKU/store combos to the appropriate forecasting model based on demand characteristics
# Transform once, then compute ADI and CV² from transformed stats
def transform(df: pd.DataFrame, ddof: int = 0) -> pd.DataFrame:
    return (
        df.groupby("unique_id")
        .agg(
            total_days           = ("ds", lambda s: s.count()),
            non_zero_days        = ("y",  lambda s: (s > 0).sum()),
            mean_demand_non_zero = ("y",  lambda s: s[s > 0].mean()),
            std_demand_non_zero  = ("y",  lambda s: s[s > 0].std(ddof=ddof)),
        )
        .reset_index()
    )

def compute_adi(df: pd.DataFrame) -> pd.DataFrame:
    # ADI of 1.0   means that, on average, there is one non-zero demand per period (daily in this case)
    # ADI of > 1.0 means "at least one zero-demand day exists anywhere in the series"
    return df.assign(
        adi=lambda df_: df_["total_days"] / df_["non_zero_days"].where(df_["non_zero_days"] > 0)
    )


def compute_cv2(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        cv2=lambda df_: (
            (df_["std_demand_non_zero"] / df_["mean_demand_non_zero"].where(df_["mean_demand_non_zero"] > 0)) ** 2
        )
    )
