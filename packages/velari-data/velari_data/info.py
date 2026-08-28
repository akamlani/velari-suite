import pandas as pd
import numpy as np
import scipy.stats as scs
from   typing import Optional

class InfoSchema(object):
    @staticmethod
    def _cast_datetime(series: pd.Series, unit: Optional[str] = None) -> pd.Series:
        if unit == "unix_s":
            return pd.to_datetime(series, unit="s", errors="coerce")
        if unit == "unix_ms":
            return pd.to_datetime(series, unit="ms", errors="coerce")
        return pd.to_datetime(series, errors="coerce")


class InfoText(object):
    WORDS_PER_PAGE  = 250      # standard manuscript page (double-spaced, 12pt, 1" margins)
    TOKENS_PER_WORD = 4 / 3    # OpenAI's tokenizer rule of thumb: ~0.75 words per token
    TOKENS_PER_PAGE = WORDS_PER_PAGE * TOKENS_PER_WORD
    # token-based so density (technical/jargon-heavy text tokenizes into more subword tokens per
    # word) is already priced in, no separate dense/sparse branch needed.

    @classmethod
    def calc_record_stats(cls, df: pd.DataFrame, col: str='text') -> pd.DataFrame:
        """Calculate text statistics for a specific column in a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame; must already have `words`/`tokens` columns
                (e.g. via `Tokenizer.pipe`).
            col (str, optional): Column name to calculate statistics for. Defaults to 'text'.

        Returns:
            pd.DataFrame: Input columns plus char_sz/word_sz/token_sz/page_sz per record.

        Examples:
            >>> df = pd.DataFrame({"text": ["hello world", "foo bar baz", "a"]})
            >>> stats = InfoText.calc_record_stats(df, "text")
            >>> stats.columns.tolist()
            ['text', 'char_sz', 'word_sz', 'token_sz', 'page_sz']
        """
        # can then compute global len |D| (min, max, std, avg) statistics across all records
        return df.assign(
            # could use a specific tokenizer for this
            char_sz     =  lambda df_: df_[col].apply(len),
            word_sz     =  lambda df_: df_["words"].apply(len),
            token_sz    =  lambda df_: df_["tokens"].apply(len),
        ).assign(
            page_sz     =  lambda df_: (df_["token_sz"] / cls.TOKENS_PER_PAGE).round(1),
        )

    @classmethod
    def calc_frame_stats(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate frame statistics for numeric columns in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to calculate statistics for.

        Returns:
            pd.DataFrame: Statistics for numeric columns in the DataFrame.

        # should be equivalent to form minimum:
        avg_char_sz  =  np.mean(char_sz)
        avg_word_sz  =  np.mean(word_sz)
        avg_token_sz =  np.mean(token_sz)
        """
        cols_numeric  = df.select_dtypes(include=['number']).columns
        df_base_stats =  df[cols_numeric].agg(['sum', 'min', 'max', 'std', 'mean', 'median']).round(3)
        # optimize the following for list comprehension
        df_extended_stats = pd.DataFrame({
            fn.__name__: [fn(df[col]).round(3) for col in cols_numeric]
            for fn in [scs.skew, scs.kurtosis]
        }, index=cols_numeric).T
        return pd.concat([df_base_stats, df_extended_stats], axis=0)
