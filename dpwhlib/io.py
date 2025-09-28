import io
from pathlib import Path
import pandas as pd

def read_base_csv_from_path(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def read_budget_xlsx_from_path(path: Path) -> tuple[pd.DataFrame, str]:
    xls = pd.ExcelFile(path)
    use_sheet = None
    for sh in xls.sheet_names:
        head = pd.read_excel(path, sheet_name=sh, nrows=5)
        cols = " ".join(map(str, head.columns)).lower()
        if any(k in cols for k in ["budget","amount","appropriation","gaa","nep","gab","program","project","title","location","region","province","city","municipality","year","contract","code","id"]):
            use_sheet = sh
            break
    if use_sheet is None:
        use_sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=use_sheet)
    return df, use_sheet

def save_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
