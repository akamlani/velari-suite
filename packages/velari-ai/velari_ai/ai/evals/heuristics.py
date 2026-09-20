import  numpy as np
import  re
from    collections import Counter
from    typing import List


def contains_keywords(output: str, keywords:List[str]) -> bool:
    return any(keyword.strip() in output.lower().strip() for keyword in keywords)

def count_keywords(output: str, keywords:List[str]) -> int:
    return sum(1 for keyword in keywords if keyword.strip().lower() in output.strip().lower())

def text_distribution(output: str, keywords:List[str]) -> Counter:
    words = output.strip().lower().split()
    keywords_lower = {keyword.strip().lower() for keyword in keywords}
    filtered_words = [word for word in words if word in keywords_lower]
    return Counter(filtered_words)

def contains_link(output: str) -> bool:
    pattern = r"https?://[^\s]+"
    return bool(re.search(pattern, output))

def exact_match(output: str, expected: str) -> bool:
    return output.strip() == expected.strip()

def wordiness(expected: str, output: str) -> bool:
    return len(output.split()) < len(expected.split())

def in_bounds(value: float, lower: float, upper: float) -> bool:
    return lower <= value <= upper