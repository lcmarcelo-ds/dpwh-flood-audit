import re
import numpy as np
import pandas as pd
from .textutils import norm_text, seq_ratio

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
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

def compute_flags(df: pd.DataFrame) -> dict:
    out = {}

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

    len_cols = _find_cols_any(cols, r"(length|linear|km|meters|m\.)")
    area_cols = _find_cols_any(cols, r"(area|sqkm|sqm|hect|ha)")

    df = df.copy()
    if start_col: df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    if end_col:   df[end_col]   = pd.to_datetime(df[end_col], errors="coerce")

    df["__Year"] = df[year_col].apply(_extract_year).astype("Int64") if year_col else pd.Series(pd.NA, index=df.index, dtype="Int64")
    if df["__Year"].isna().all():
        df["__Year"] = df[[c for c in [start_col, end_col] if c]].apply(
            lambda r: r.dropna().iloc[0].year if r.dropna().shape[0]>0 else np.nan, axis=1
        ).astype("Int64")

    if amount_col:
        df["__Amount"] = pd.to_numeric(df[amount_col].astype(str).str.replace(",","", regex=False), errors="coerce")
    else:
        df["__Amount"] = np.nan

    loc_cols = [c for c in [region_col, province_col, city_col, barangay_col] if c]
    if loc_cols:
        df["__AreaKey"] = df[loc_cols].astype(str).agg(", ".join, axis=1).map(norm_text)
    else:
        loc_guess = _find_col(cols, r"(location|site|river|barangay|place)")
        df["__AreaKey"] = df[loc_guess].astype(str).map(norm_text) if loc_guess else ""

    if start_col and end_col:
        df["__DurationDays"] = (df[end_col] - df[start_col]).dt.days
    elif start_col:
        df["__DurationDays"] = (pd.Timestamp.now() - df[start_col]).dt.days
    else:
        df["__DurationDays"] = np.nan

    if title_col:
        df["__TitleNorm"] = df[title_col].astype(str).map(norm_text)
    else:
        df["__TitleNorm"] = ""

    length_col = len_cols[0] if len_cols else None
    area_col = area_cols[0] if area_cols else None

    if length_col:
        df["__LengthKm"] = pd.to_numeric(df[length_col].astype(str).str.replace(",","", regex=False), errors="coerce")
        med = df["__LengthKm"].dropna().median() if df["__LengthKm"].notna().any() else np.nan
        if pd.notna(med) and med > 1000:
            df["__LengthKm"] = df["__LengthKm"] / 1000.0
    else:
        df["__LengthKm"] = np.nan

    if area_col:
        avals = pd.to_numeric(df[area_col].astype(str).str.replace(",","", regex=False), errors="coerce")
        med = avals.dropna().median() if not avals.dropna().empty else np.nan
        if pd.notna(med) and med > 100000:
            df["__AreaSqKm"] = avals / 1_000_000.0
        elif pd.notna(med) and 1 <= med <= 10000:
            df["__AreaSqKm"] = avals * 0.01
        else:
            df["__AreaSqKm"] = avals
    else:
        df["__AreaSqKm"] = np.nan

    df["__CostPerKm"]   = df.apply(lambda r: _safe_div(r["__Amount"], r["__LengthKm"]) if pd.notna(r["__LengthKm"]) else np.nan, axis=1)
    df["__CostPerSqKm"] = df.apply(lambda r: _safe_div(r["__Amount"], r["__AreaSqKm"]) if pd.notna(r["__AreaSqKm"]) else np.nan, axis=1)

    # Redundant
    redundant_idx = set()
    if title_col:
        grp = df.groupby(["__AreaKey", "__Year"], dropna=False)
        for _, sub in grp:
            if len(sub) < 2: continue
            titles = sub[title_col].astype(str).tolist()
            idxs = sub.index.tolist()
            for i in range(len(titles)):
                for j in range(i+1, len(titles)):
                    if seq_ratio(titles[i], titles[j]) >= 0.60:
                        redundant_idx.add(idxs[i]); redundant_idx.add(idxs[j])
    df["FLAG_RedundantSameAreaYear"] = df.index.isin(redundant_idx)

    # Potential Ghost
    amt_hi = np.nanpercentile(df["__Amount"].dropna(), 75) if df["__Amount"].notna().any() else np.nan
    flags_g = []; reasons_g = []
    for _, r in df.iterrows():
        status = str(r.get(status_col, "")).lower().strip() if status_col else ""
        sd = r.get(start_col); ed = r.get(end_col); amt = r.get("__Amount")
        flag=False; reasons=[]
        if ("complete" in status) or ("100" in status):
            if pd.isna(ed):
                flag=True; reasons.append("Completed status but no completion date")
            elif pd.notna(sd) and (ed - sd).days < 7 and pd.notna(amt) and pd.notna(amt_hi) and amt >= amt_hi:
                flag=True; reasons.append("Very short completion for high-amount project")
        if pd.notna(sd) and pd.isna(ed) and (pd.Timestamp.now() - sd).days > 730:
            flag=True; reasons.append(">2 years elapsed without completion")
        if pd.isna(sd) and pd.isna(ed) and pd.notna(amt) and pd.notna(amt_hi) and amt >= amt_hi:
            flag=True; reasons.append("No dates recorded for high-amount project")
        flags_g.append(flag); reasons_g.append("; ".join(reasons))
    df["FLAG_PotentialGhost"] = flags_g
    df["Reason_PotentialGhost"] = reasons_g

    # Never-ending
    flags_n=[]; reasons_n=[]
    for _, r in df.iterrows():
        flag=False; reasons=[]
        dur = r["__DurationDays"]
        if pd.notna(dur) and dur >= 730:
            flag=True; reasons.append(f"Duration {int(dur)} days (>=730)")
        if title_col:
            sub = df[(df["__AreaKey"]==r["__AreaKey"]) & df[title_col].notna()]
            years_hit=set()
            for y, suby in sub.groupby("__Year"):
                if pd.isna(y): continue
                if suby[title_col].astype(str).apply(lambda t: seq_ratio(t, r[title_col])).max() >= 0.60:
                    years_hit.add(int(y))
            if len(years_hit)>=3:
                flag=True; reasons.append(f"Similar title across {len(years_hit)} years in same area")
        flags_n.append(flag); reasons_n.append("; ".join(reasons))
    df["FLAG_NeverEnding"] = flags_n
    df["Reason_NeverEnding"] = reasons_n

    # Costly
    def iqr_flags(s: pd.Series, k=1.5):
        s = s.astype(float)
        s2 = s[~s.isna()]
        if s2.empty:
            return pd.Series(False, index=s.index), np.nan, np.nan
        q1, q3 = s2.quantile(0.25), s2.quantile(0.75)
        iqr = q3-q1 if (q3-q1)>0 else (s2.std() if s2.std()>0 else 1.0)
        low, high = q1 - k*iqr, q3 + k*iqr
        flags = (s<low)|(s>high)
        return flags.fillna(False), low, high

    metric = "__CostPerKm" if df["__CostPerKm"].notna().sum() >= df["__CostPerSqKm"].notna().sum() else "__CostPerSqKm"
    flags_c, low_t, high_t = iqr_flags(df[metric])
    df["FLAG_Costly"] = flags_c
    df["CostMetricUsed"] = metric
    df["CostOutlierLow"], df["CostOutlierHigh"] = low_t, high_t

    flag_cols = ["FLAG_RedundantSameAreaYear","FLAG_PotentialGhost","FLAG_NeverEnding","FLAG_Costly"]
    out["annotated_full"] = df.copy()
    out["redundant"] = df[df["FLAG_RedundantSameAreaYear"]].copy()
    out["ghost"] = df[df["FLAG_PotentialGhost"]].copy()
    out["neverending"] = df[df["FLAG_NeverEnding"]].copy()
    out["costly"] = df[df["FLAG_Costly"]].copy()
    out["all_flagged"] = df[df[flag_cols].any(axis=1)].copy()

    out["detected_columns"] = pd.DataFrame({
        "role": ["amount","title","status","start","end","year","region","province","city/municipality","barangay","length_cols","area_cols"],
        "column": [amount_col,title_col,status_col,start_col,end_col,year_col,region_col,province_col,city_col,barangay_col,
                   (len_cols[0] if len_cols else None),(area_cols[0] if area_cols else None)]
    })

    out["summary"] = pd.DataFrame({
        "Flag": ["RedundantSameAreaYear","PotentialGhost","NeverEnding","Costly","AnyFlag"],
        "Count": [len(out["redundant"]),len(out["ghost"]),len(out["neverending"]),len(out["costly"]),len(out["all_flagged"])]
    })
    return out
