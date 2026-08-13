import numpy as np
import re
from   collections.abc import Iterable, Iterator
from   itertools import batched, islice
from   typing import List, Tuple, TypeVar, Sequence

T = TypeVar("T")

# remove citation references like [1], [2], etc.
trsfrm_citation  = lambda text: re.sub(r"\[\d+\]", "", text)
# remove non-alphanumeric characters and replace spaces with underscores
trsfrm_text      = lambda text: re.sub(r'[-\s]+', '_', re.sub(r'[^\w\s-]', '', text))
trsfrm_textlines = lambda texts: [text.replace("\n", " ").strip() for text in texts if text.strip()]

class Generator(object):
    @staticmethod
    def batchgen_delegate(iterable: Iterable[T], batch_size: int) -> Iterator[Tuple[T, ...]]:
        return batched(iterable, batch_size)

    @staticmethod
    def batchgen_iterable(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
        it = iter(iterable)
        for first in it:
            yield [first, *islice(it, batch_size - 1)]

    @staticmethod
    def batchgen_sequence(sequence: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
        for start in range(0, len(sequence), batch_size):
            yield sequence[start:start + batch_size]

    @staticmethod
    def chunk_text(text: str, chunk_size: int=300, overlap: int=50) -> list[str]:
        words  = text.split()
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size - overlap)
        ]


def trsfrm_normalize_vector(x: np.ndarray) -> np.ndarray:
    """Normalize a vector to have unit length, e.g., for embeddings."""
    norm = np.linalg.norm(x)
    return x / norm if norm != 0 else x

def trsfrm_normalizer(x: np.ndarray) -> np.ndarray:
    return ((x - np.min(x)) / (np.max(x) - np.min(x))
           if np.max(x) != np.min(x) else np.zeros_like(x))
