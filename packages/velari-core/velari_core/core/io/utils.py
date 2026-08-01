import pandas as pd
import re

import base64

trsfrm_base64 = lambda data: base64.b64encode(data).decode("utf-8")


def trsfrm_camelcase_to_snakecase(col: str) -> str:
    """Transforms column naming from camelcase to snakecase

    Args:
        col (str): input column name to transfrom

    Returns:
        str: transformed column
    """
    #  column = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    column = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", col)
    column = re.sub("([a-z0-9])([A-Z])", r"\1_\2", col).lower()
    return column.replace(" ", "_")


def trsfrm_frame_camelcase_to_snakecase(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms column naming from camelcase to snakecase for a dataframe

    Args:
        df (pd.DataFrame): input dataframe with columns

    Returns:
        pd.DataFrame: transformed pandas dataframe
    """
    df.columns = list(map(trsfrm_camelcase_to_snakecase, df.columns))
    return df
