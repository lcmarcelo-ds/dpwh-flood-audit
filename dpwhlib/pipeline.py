import pandas as pd
from .flags import compute_flags

def compute_all_flags(df: pd.DataFrame) -> dict:
    return compute_flags(df)
