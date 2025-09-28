from pathlib import Path
import io, csv
import pandas as pd

COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
COMMON_SEPS = [",", ";", "\t", "|"]

def _sniff_sep(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(COMMON_SEPS))
        return dialect.delimiter
    except Exception:
        counts = {sep: sample.count(sep) for sep in COMMON_SEPS}
        return max(counts, key=counts.get) if any(counts.values()) else ","

def read_base_csv_from_path(path: Path) -> pd.DataFrame:
    """
    Robust CSV loader:
      • Tries multiple encodings
      • Sniffs delimiter
      • Skips bad lines
      • Falls back to read_excel if 'csv' parse fails
    """
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"{path} is missing or empty.")

    raw = path.read_bytes()

    # sniff delimiter from first ~32KB
    head_bytes = raw[:32768]
    try:
        head_text = head_bytes.decode("utf-8", errors="ignore")
    except Exception:
        head_text = head_bytes.decode("latin-1", errors="ignore")
    sep = _sniff_sep(head_text)

    last_err = None
    for enc in COMMON_ENCODINGS:
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                sep=sep,
                engine="python",
                encoding=enc,
                on_bad_lines="skip"
            )
        except Exception as e:
            last_err = e

    # fallback: file might actually be Excel
    try:
        return pd.read_excel(io.BytesIO(raw))
    except Exception:
        pass

    raise last_err if last_err else ValueError("Unable to parse file; unknown format/encoding.")

def save_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
