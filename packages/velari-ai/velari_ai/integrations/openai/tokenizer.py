import pandas as pd
import tiktoken
from typing import List


class Tokenizer(object):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._enc = tiktoken.encoding_for_model(model)

    @property
    def encoding_name(self) -> str:
        return self._enc.name

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._enc.decode(tokens)

    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self._enc.encode(text)
        return self._enc.decode(tokens[:max_tokens])

    def count_col(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df.assign(**{f"{col}_token_cnt": df[col].map(self.count, na_action="ignore")})

    def encode_col(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df.assign(**{f"{col}_tokens": df[col].map(self.encode, na_action="ignore")})

    def truncate_col(self, df: pd.DataFrame, col: str, max_tokens: int) -> pd.DataFrame:
        return df.assign(
            **{f"{col}_truncated": df[col].map(lambda text: self.truncate(text, max_tokens), na_action="ignore")}
        )

    def pipe(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df.assign(
            **{
                "words":   df[col].map(str.split, na_action="ignore"),
                "tokens":  df[col].map(self.encode, na_action="ignore"),
                "decoded": df[col].map(lambda text: self.decode(self.encode(text)), na_action="ignore"),
            }
        )

    @staticmethod
    def get_encodings_available() -> List[str]:
        # ['gpt2', 'r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base', 'o200k_base']
        return tiktoken.list_encoding_names()

    @staticmethod
    def get_models_info() -> pd.DataFrame:
        from tiktoken.model import MODEL_TO_ENCODING

        df = pd.DataFrame(MODEL_TO_ENCODING.items(), columns=["model", "encoding"])
        enc_meta = {name: tiktoken.get_encoding(name) for name in df["encoding"].unique()}
        return (
            df.assign(
                n_vocab=df["encoding"].map(lambda e: enc_meta[e].n_vocab),
                eot_token=df["encoding"].map(lambda e: enc_meta[e].eot_token),
            )
            .sort_values("model")
            .reset_index(drop=True)
        )
