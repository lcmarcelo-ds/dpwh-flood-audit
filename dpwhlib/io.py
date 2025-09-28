from pathlib import Path
import io
import pandas as pd

def read_base_csv_from_path(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
