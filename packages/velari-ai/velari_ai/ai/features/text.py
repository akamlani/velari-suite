import re
from   typing import List

def trsfrm_text_to_words(text: str) -> List[str]:
    # lowercase, keep simple words
    return re.findall(r"[a-z']+", text.lower())

def approximate_token_count(text: str) -> int:
    """Crude estimate of the number of tokens ~ 4 characters per token."""
    return max(1, round(len(text) / 4))
