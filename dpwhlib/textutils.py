import re
from difflib import SequenceMatcher

STOPWORDS = {"of","the","and","for","to","in","a","an"}

def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s\-/,_()&.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_set(s: str) -> set[str]:
    return {t for t in re.split(r"[\s\-/,_()&.]+", norm_text(s)) if t and t not in STOPWORDS}

def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()

def partial_ratio(a: str, b: str) -> float:
    a = norm_text(a); b = norm_text(b)
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    best = 0.0
    for start in range(0, max(1, len(b) - len(a) + 1)):
        sub = b[start:start+len(a)]
        r = SequenceMatcher(None, a, sub).ratio()
        if r > best:
            best = r
        if best == 1.0:
            break
    return best

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter/union if union else 0.0

def prefix_overlap(a: str, b: str) -> float:
    a = norm_text(a); b = norm_text(b)
    n = min(len(a), len(b))
    k = 0
    for i in range(n):
        if a[i] == b[i]:
            k += 1
        else:
            break
    return k/float(n) if n > 0 else 0.0
