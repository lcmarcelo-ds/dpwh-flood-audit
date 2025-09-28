import re

def norm_text(s: str) -> str:
    s = "" if s is None else str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s\-/,_()&.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
