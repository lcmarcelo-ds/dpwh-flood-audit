# dpwhlib/flags.py
import re
import warnings
import numpy as np
import pandas as pd
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
from .textutils import norm_text

# ---------- helpers ----------
def _find_col(cols, patterns):
    for c in cols:
        if re.search(patterns, str(c), re.I):
            return c
    return None

def _find_cols_any(cols, patterns):
    return [c for c in cols if re.search(patterns, str(c), re.I)]

def _extract_year(val):
    s = str(val)
    m = re.search(r"(20\d{2}|19\d{2})", s)
    return int(m.group(1)) if m else np.nan

def _safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

# ---------- stage 1: one-time preprocessing (cacheable) ----------
def preprocess_projects(df_raw: pd.DataFrame) -> dict:
    """
    Heavy work done once:
      - detect columns
      - parse dates quietly
      - compute helper columns (__Year, __Amount, __AreaKey, __DurationDays, __LengthKm, __AreaSqKm, __CostPer*)
      - normalized strings for fast matching (__TitleNorm, __ContractorNorm)
    Returns dict with 'prepped' (DataFrame) and 'colmap' (detected columns).
    """
    df = df_raw.copy()
    cols = list(df.columns)

    title_col  = _find_col(cols, r"(project|title|scope|description|name)")
    amount_col = _find_col(cols, r"(amount|contract|abc|cost|budget)")
    status_col = _find_col(cols, r"(status|physical|percent|progress)")
    start_col  = _find_col(cols, r"(start|commence|ntp|notice to proceed|date started)")
    end_col    = _find_col(cols, r"(end|completion|date completed|target)")
    year_col   = _find_col(cols, r"\byear\b")
    region_col   = _find_col(cols, r"\bregion\b")
    province_col = _find_col(cols, r"\bprovince\b")
    city_col     = _find_col(cols, r"(city|municipality|muni|lgu)")
    barangay_col = _find_col(cols, r"(barangay|brgy)")
    contractor_col = _find_col(cols, r"(contractor|supplier|winning\s*bidder|provider|vendor)")

    len_cols = _find_cols_any(cols, r"(length|linear|km|meters|m\.)")
    area_cols = _find_cols_any(cols, r"(area|sqkm|sqm|hect|ha)")

    # Quiet mixed-date warnings and coerce invalids to NaT
    if start_col:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[start_col] = pd.to_datetime(df[start_col], errors="coerce", format="mixed")
    if end_col:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[end_col] = pd.to_datetime(df[end_col], errors="coerce", format="mixed")

    # Year
    if year_col:
        df["__Year"] = df[year_col].apply(_extract_year).astype("Int64")
    else:
        df["__Year"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        use = [c for c in [start_col, end_col] if c]
        if use:
            df["__Year"] = df[use].apply(lambda r: r.dropna().iloc[0].year if r.dropna().shape[0]>0 else np.nan, axis=1).astype("Int64")

    # Amount
    if amount_col:
        df["__Amount"] = pd.to_numeric(df[amount_col].astype(str).str.replace(",","", regex=False), errors="coerce")
    else:
        df["__Amount"] = np.nan

    # Area key
    loc_cols = [c for c in [region_col, province_col, city_col, barangay_col] if c]
    if loc_cols:
        df["__AreaKey"] = df[loc_cols].astype(str).agg(", ".join, axis=1).map(norm_text)
    else:
        loc_guess = _find_col(cols, r"(location|site|river|barangay|place)")
        df["__AreaKey"] = df[loc_guess].astype(str).map(norm_text) if loc_guess else ""

    # Duration
    if start_col and end_col:
        df["__DurationDays"] = (df[end_col] - df[start_col]).dt.days
    elif start_col:
        df["__DurationDays"] = (pd.Timestamp.now() - df[start_col]).dt.days
    else:
        df["__DurationDays"] = np.nan

    # Norm strings
    df["__TitleNorm"] = df[title_col].astype(str).map(norm_text) if title_col else ""
    df["__ContractorNorm"] = df[contractor_col].astype(str).map(norm_text) if contractor_col else ""

    # Unit conversions
    length_col = len_cols[0] if len_cols else None
    area_col   = area_cols[0] if area_cols else None

    if length_col:
        xlen = pd.to_numeric(df[length_col].astype(str).str.replace(",","",regex=False), errors="coerce")
        if xlen.dropna().median() > 1000:
            xlen = xlen/1000.0
        df["__LengthKm"] = xlen
    else:
        df["__LengthKm"] = np.nan

    if area_col:
        avals = pd.to_numeric(df[area_col].astype(str).str.replace(",","",regex=False), errors="coerce")
        med = avals.dropna().median() if avals.notna().any() else np.nan
        if pd.notna(med) and med > 100000:
            avals = avals/1_000_000.0
        elif pd.notna(med) and 1 <= med <= 10000:
            avals = avals*0.01
        df["__AreaSqKm"] = avals
    else:
        df["__AreaSqKm"] = np.nan

    df["__CostPerKm"]   = df.apply(lambda r: _safe_div(r["__Amount"], r["__LengthKm"]) if pd.notna(r["__LengthKm"]) else np.nan, axis=1)
    df["__CostPerSqKm"] = df.apply(lambda r: _safe_div(r["__Amount"], r["__AreaSqKm"]) if pd.notna(r["__AreaSqKm"]) else np.nan, axis=1)

    colmap = {
        "title": title_col, "amount": amount_col, "status": status_col,
        "start": start_col, "end": end_col, "year": year_col,
        "region": region_col, "province": province_col, "city": city_col, "barangay": barangay_col,
        "contractor": contractor_col, "length": length_col, "area": area_col
    }
    return {"prepped": df, "colmap": colmap}

# ---------- stage 2: fast flags from prepped df ----------
def _fast_redundant_flags(prepped: pd.DataFrame, title_col: str, similarity: float) -> pd.Series:
    """
    Mark projects as redundant if there exists another in the same (AreaKey, Year)
    with token_set_ratio >= similarity threshold.
    Uses rapidfuzz.cdist (C-optimized) per block.
    """
    if not title_col:
        return pd.Series(False, index=prepped.index)

    thr = int(round(similarity * 100))  # rapidfuzz returns 0-100
    flags = np.zeros(len(prepped), dtype=bool)

    # Work in blocks to avoid O(n^2) across whole dataset
    for (_, _), sub in prepped.groupby(["__AreaKey", "__Year"], dropna=False):
        if len(sub) < 2:
            continue
        titles = sub[title_col].astype(str).tolist()
        # Compute pairwise token_set_ratio (fast C) with a cutoff
        mat = rf_process.cdist(
            titles, titles,
            scorer=rf_fuzz.token_set_ratio,
            score_cutoff=thr
        )
        # mat is dense; mark i if any j != i has score >= thr
        # We'll look only at upper triangle to avoid diagonal/self
        n = len(titles)
        hit_rows = set()
        for i in range(n):
            # Check any j where mat[i,j] >= thr and j!=i
            # Avoid scanning all cells: slice & mask
            row = mat[i]
            # rapidfuzz returns numpy array, diagonal equals 100, so we need > thr or >= thr with j!=i
            if (row[:i] >= thr).any() or (row[i+1:] >= thr).any():
                hit_rows.add(i)
        if hit_rows:
            flags[sub.index[list(hit_rows)]] = True
    return pd.Series(flags, index=prepped.index)

def compute_project_flags_fast(
    prepped: pd.DataFrame,
    colmap: dict,
    redundant_similarity=0.60,
    ghost_high_amount_percentile=75,
    never_ending_days=730,
    cost_iqr_k=1.5
) -> dict:
    df = prepped.copy()
    title_col   = colmap.get("title")
    amount_col  = colmap.get("amount")
    status_col  = colmap.get("status")
    start_col   = colmap.get("start")
    end_col     = colmap.get("end")

    # Redundant (fast)
    df["FLAG_RedundantSameAreaYear"] = _fast_redundant_flags(df, title_col, redundant_similarity)

    # Potential Ghost
    amt_hi = np.nanpercentile(df["__Amount"].dropna(), ghost_high_amount_percentile) if df["__Amount"].notna().any() else np.nan
    status_vals = df[status_col].astype(str).str.lower().str.strip() if status_col else pd.Series("", index=df.index)
    completeish = status_vals.str.contains("complete") | status_vals.str.contains(r"\b100\b")
    no_end = df[end_col].isna() if end_col else pd.Series(True, index=df.index)
    very_short = (
        (df[end_col] - df[start_col]).dt.days < 7
        if start_col and end_col else pd.Series(False, index=df.index)
    )
    high_amt = (df["__Amount"] >= amt_hi) if np.isfinite(amt_hi) else pd.Series(False, index=df.index)
    long_open = (
        (pd.Timestamp.now() - df[start_col]).dt.days > 730
        if start_col else pd.Series(False, index=df.index)
    )
    no_dates = (df[start_col].isna() if start_col else True) & (df[end_col].isna() if end_col else True)
    ghost_flag = (
        (completeish & no_end) |
        (completeish & very_short & high_amt) |
        (long_open & no_end) |
        (no_dates & high_amt)
    )
    reasons = []
    # Reasons are lightweight to compute; keep clarity
    reasons.append(np.where(completeish & no_end, "Completed status but no completion date", ""))
    reasons.append(np.where(completeish & very_short & high_amt, "Very short completion for high-amount project", ""))
    reasons.append(np.where(long_open & no_end, ">2 years elapsed without completion", ""))
    reasons.append(np.where(no_dates & high_amt, "No dates recorded for high-amount project", ""))
    reason_col = pd.Series(["; ".join([r for r in row if r]) for row in zip(*reasons)], index=df.index)

    df["FLAG_PotentialGhost"] = ghost_flag.fillna(False)
    df["Reason_PotentialGhost"] = reason_col

    # Never-ending
    dur_ok = df["__DurationDays"].fillna(-1)
    df["FLAG_NeverEnding"] = dur_ok.ge(never_ending_days)

    # Costly (IQR)
    def iqr_bounds(s: pd.Series, k=1.5):
        s2 = s.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if s2.empty:
            return np.nan, np.nan
        q1, q3 = s2.quantile(0.25), s2.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            # fallback to std if needed
            std = s2.std()
            iqr = std if std > 0 else 1.0
        return (q1 - k*iqr, q3 + k*iqr)

    metric_name = "__CostPerKm" if df["__CostPerKm"].notna().sum() >= df["__CostPerSqKm"].notna().sum() else "__CostPerSqKm"
    low_t, high_t = iqr_bounds(df[metric_name], k=cost_iqr_k)
    costly = (df[metric_name] < low_t) | (df[metric_name] > high_t) if np.isfinite(low_t) and np.isfinite(high_t) else pd.Series(False, index=df.index)
    df["FLAG_Costly"] = costly.fillna(False)
    df["CostMetricUsed"] = metric_name
    df["CostOutlierLow"], df["CostOutlierHigh"] = low_t, high_t

    # Outputs
    flag_cols = ["FLAG_RedundantSameAreaYear","FLAG_PotentialGhost","FLAG_NeverEnding","FLAG_Costly"]
    out = {
        "annotated_full": df.copy(),
        "redundant": df[df["FLAG_RedundantSameAreaYear"]].copy(),
        "ghost": df[df["FLAG_PotentialGhost"]].copy(),
        "neverending": df[df["FLAG_NeverEnding"]].copy(),
        "costly": df[df["FLAG_Costly"]].copy(),
        "all_flagged": df[df[flag_cols].any(axis=1)].copy(),
        "summary": pd.DataFrame({
            "Flag": ["RedundantSameAreaYear","PotentialGhost","NeverEnding","Costly","AnyFlag"],
            "Count": [
                int(df["FLAG_RedundantSameAreaYear"].sum()),
                int(df["FLAG_PotentialGhost"].sum()),
                int(df["FLAG_NeverEnding"].sum()),
                int(df["FLAG_Costly"].sum()),
                int(df[flag_cols].any(axis=1).sum())
            ]
        })
    }
    return out

# --- Backward-compat shim: keep old API name working ---
def compute_project_flags(
    df,
    redundant_similarity=0.60,
    ghost_high_amount_percentile=75,
    never_ending_days=730,
    cost_iqr_k=1.5
):
    """
    Legacy wrapper for backward compatibility.
    Runs the new two-stage pipeline under the hood.
    """
    prep = preprocess_projects(df)
    return compute_project_flags_fast(
        prep["prepped"],
        prep["colmap"],
        redundant_similarity=redundant_similarity,
        ghost_high_amount_percentile=ghost_high_amount_percentile,
        never_ending_days=never_ending_days,
        cost_iqr_k=cost_iqr_k,
    )
