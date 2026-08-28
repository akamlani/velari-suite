import  pandas as pd
from    rich.console import Console
from    rich.table   import Table


class DisplayConsole(object):
    """Render a pandas DataFrame as a rich Table.

    Args:
        console (Console): Target console to print the rendered table to.
    """
    def __init__(self, console: Console) -> None:
        self._console = console

    def print_frame(self, df: pd.DataFrame, title: str) -> None:
        table = Table(title=title, expand=True)
        for col in df.columns:
            table.add_column(str(col))
        for row in df.itertuples(index=False):
            table.add_row(*(str(v) for v in row))
        self._console.print(table)

    def truncate_column(self, df: pd.DataFrame, column: str, max_chars: int) -> pd.DataFrame:
        """Truncate a text column to at most max_chars, appending '...' when actually truncated.

        Args:
            df (pd.DataFrame): Source frame; not mutated.
            column (str): Column to truncate.
            max_chars (int): Maximum characters to keep before appending '...'.

        Returns:
            pd.DataFrame: Copy of `df` with `column` truncated; NaN values pass through unchanged.
        """
        return df.assign(**{
            column: df[column].map(
                lambda text: text if pd.isna(text) or len(text) <= max_chars else text[:max_chars] + "..."
            )
        })
