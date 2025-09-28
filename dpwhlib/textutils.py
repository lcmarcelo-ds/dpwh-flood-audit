import re
from difflib import SequenceMatcher

STOPWORDS = {"of","the","and","for","to","in","a","an"}

def norm_text(s: str) -> str:
    s = "" if s is None else str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s\-/,_()&.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()

