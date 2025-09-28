# dpwhlib/flags.py
import re
import warnings
import numpy as np
import pandas as pd
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

# ----------------- helpers -----------------
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

def _norm_text(s: str) -> str:
    s = "" if s is None else str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s\-/,_()&.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- tokens / generic phrase control for redundancy ---
GENERIC_PHRASES = {
    "construction of", "rehabilitation of", "improvement of", "repair of",
    "construction", "rehabilitation", "improvement", "repair",
    "flood control", "flood control structure", "river control", "slope protection",
    "revetment", "desilting", "dredging", "riprap", "drainage"
}
STOP_TOKENS = {
    "the","of","and","for","to","in","a","an","phase","package","lot",
    "section","stage","barangay","brgy","city","municipality","province",
    "region","lgu","dpwh","river","creek","canal","drainage"
}
def _tokenize(s: str):
    s = _norm_text(s)
    toks = [t for t in re.split(r"[^\w]+", s) if t]
    return [t for t in toks if t not in STOP_TOKENS]

def _strip_generic_phrases(s: str) -> str:
    s2 = _norm_text(s)
    for p in GENERIC_PHRASES:
        s2 = s2.replace(p, " ")
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

# ----------------- stage 1: preprocess (cacheable) -----------------
def preprocess_projects(df_raw: pd.DataFrame, overrides: dict | None = None) -> dict:
    df = df_raw.copy()
    cols = list(df.columns)
    overrides = overrides or {}

    title_col  = overrides.get("title")  or _find_col(cols, r"(project|title|scope|description|name)")
    amount_col = overrides.get("amount") or _find_col(cols, r"(amount|contract.*amount|abc|cost|budget|approved.*budget|project.*cost)")
    status_col = overrides.get("status") or _find_col(cols, r"(status|physical|accomplish|progress|%|percent)")
    start_col  = overrides.get("start")  or _find_col(cols, r"(start|commence|ntp|notice\s*to\s*proceed|date\s*started|date\s*start)")
    end_col    = overrides.get("end")    or _find_col(cols, r"(end|completion|date\s*completed|actual\s*completion)")
    target_col = overrides.get("target") or _find_col(cols, r"(target\s*completion|target\s*date)")
    year_col   = overrides.get("year")   or _find_col(cols, r"\byear\b")
    region_col   = overrides.get("region")   or _find_col(cols, r"\bregion\b")
    province_col = overrides.get("province") or _find_col(cols, r"\bprovince\b")
    city_col     = overrides.get("city")     or _find_col(cols, r"(city|municipality|muni|lgu)")
    barangay_col = overrides.get("barangay") or _find_col(cols, r"(barangay|brgy)")
    contractor_col = overrides.get("contractor") or _find_col(cols, r"(contractor|supplier|winning\s*bidder|provider|vendor)")

    len_cols = [overrides.get("length")] if overrides.get("length") else _find_cols_any(cols, r"(\blength\b|linear\s*(m|meter|metre|lm|km)|\bkm\b|\bm\b)")
    area_cols = [overrides.get("area")]   if overrides.get("area")   else _find_cols_any(cols, r"(area|sq.?km|sqm|m2|hect|ha)")

    # Parse dates quietly
    for c in [start_col, end_col, target_col]:
        if c:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")

    # Year
    if year_col:
        df["__Year"] = df[year_col].apply(_extract_year).astype("Int64")
    else:
        df["__Year"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        use = [c for c in [start_col, end_col, target_col] if c]
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
        df["__AreaKey"] = df[loc_cols].astype(str).agg(", ".join, axis=1).map(_norm_text)
    else:
        loc_guess = _find_col(cols, r"(location|site|river|barangay|place)")
        df["__AreaKey"] = df[loc_guess].astype(str).map(_norm_text) if loc_guess else ""

    # Duration
    if start_col and end_col:
        df["__DurationDays"] = (df[end_col] - df[start_col]).dt.days
    elif start_col:
        df["__DurationDays"] = (pd.Timestamp.now() - df[start_col]).dt.days
    else:
        df["__DurationDays"] = np.nan

    # Norm strings
    df["__TitleNorm"] = df[title_col].astype(str).map(_norm_text) if title_col else ""
    df["__ContractorNorm"] = df[contractor_col].astype(str).map(_norm_text) if contractor_col else ""

    # Length/Area → km/sqkm
    length_col = len_cols[0] if len_cols and len_cols[0] else None
    area_col   = area_cols[0] if area_cols and area_cols[0] else None

    if length_col:
        xlen = pd.to_numeric(df[length_col].astype(str).str.replace(",","",regex=False), errors="coerce")
        if xlen.dropna().median() > 1000:  # meters → km
            xlen = xlen / 1000.0
        df["__LengthKm"] = xlen
    else:
        df["__LengthKm"] = np.nan

    if area_col:
        avals = pd.to_numeric(df[area_col].astype(str).str.replace(",","",regex=False), errors="coerce")
        med = avals.dropna().median() if avals.notna().any() else np.nan
        if pd.notna(med) and med > 100000:   # sqm → sqkm
            avals = avals / 1_000_000.0
        elif pd.notna(med) and 1 <= med <= 10000:  # hectares → sqkm
            avals = avals * 0.01
        df["__AreaSqKm"] = avals
    else:
        df["__AreaSqKm"] = np.nan

    df["__CostPerKm"]   = df.apply(lambda r: _safe_div(r["__Amount"], r["__LengthKm"]) if pd.notna(r["__LengthKm"]) else np.nan, axis=1)
    df["__CostPerSqKm"] = df.apply(lambda r: _safe_div(r["__Amount"], r["__AreaSqKm"]) if pd.notna(r["__AreaSqKm"]) else np.nan, axis=1)

    # Try numeric "% complete"
    if status_col:
        s = df[status_col].astype(str).str.extract(r"(\d{1,3}(?:\.\d+)?)")[0]
        df["__AccompPct"] = pd.to_numeric(s, errors="coerce").clip(0,100)
    else:
        df["__AccompPct"] = np.nan

    colmap = {
        "title": title_col, "amount": amount_col, "status": status_col,
        "start": start_col, "end": end_col, "target": target_col, "year": year_col,
        "region": region_col, "province": province_col, "city": city_col, "barangay": barangay_col,
        "contractor": contractor_col, "length": length_col, "area": area_col
    }
    return {"prepped": df, "colmap": colmap}

# ----------------- Redundant (stricter + labeled) -----------------
def _fast_redundant_flags_labeled(prepped: pd.DataFrame, title_col: str, similarity: float):
    n = len(prepped)
    flags = np.zeros(n, dtype=bool)
    group_id = pd.Series(pd.NA, index=prepped.index, dtype="Int64")
    peers = [[] for _ in range(n)]
    reasons = [""] * n

    if not title_col:
        return pd.Series(flags, index=prepped.index), group_id, pd.Series(peers), pd.Series(reasons)

    thr = int(round(similarity * 100))
    gid = 0
    idx_all = list(prepped.index)

    for (_, _), sub in prepped.groupby(["__AreaKey", "__Year"], dropna=False):
        if len(sub) < 2:
            continue
        idx = sub.index.tolist()
        titles = sub[title_col].astype(str).tolist()
        titles_stripped = [_strip_generic_phrases(t) for t in titles]
        toks = [set(_tokenize(t)) for t in titles_stripped]

        mat = rf_process.cdist(titles_stripped, titles_stripped, scorer=rf_fuzz.token_set_ratio, score_cutoff=thr)

        adj = {i: set() for i in range(len(idx))}
        for i in range(len(idx)):
            row = mat[i]
            for j in range(len(idx)):
                if j == i: 
                    continue
                if row[j] >= thr:
                    common = toks[i].intersection(toks[j])
                    if len([t for t in common if t not in GENERIC_PHRASES and len(t) > 2]) >= 2:
                        adj[i].add(j)

        seen = set()
        for i in range(len(idx)):
            if i in seen or not adj[i]:
                continue
            comp = set([i]); queue = [i]; seen.add(i)
            while queue:
                u = queue.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v); comp.add(v); queue.append(v)
            if len(comp) >= 2:
                gid += 1
                comp_idx = [idx[k] for k in sorted(list(comp))]
                for k in comp:
                    rowid = idx[k]
                    flags[idx_all.index(rowid)] = True
                    group_id.loc[rowid] = gid
                    peer_titles = [titles[m] for m in comp if m != k]
                    peers[idx_all.index(rowid)] = peer_titles
                    reasons[idx_all.index(rowid)] = f"Similar title(s) within same area-year; {len(comp)} in group"
    return pd.Series(flags, index=prepped.index), group_id, pd.Series(peers, index=prepped.index), pd.Series(reasons, index=prepped.index)

# ----------------- stage 2: compute flags fast -----------------
def compute_project_flags_fast(
    prepped: pd.DataFrame,
    colmap: dict,
    redundant_similarity=0.70,
    ghost_high_amount_percentile=75,
    never_ending_days=730,
    cost_iqr_k=1.5,
    use_target_overrun=True,
    grace_days=60
) -> dict:
    df = prepped.copy()
    title_col   = colmap.get("title")
    amount_col  = colmap.get("amount")
    status_col  = colmap.get("status")
    start_col   = colmap.get("start")
    end_col     = colmap.get("end")
    target_col  = colmap.get("target")

    # ---- Redundant (same area & same year) with labels ----
    rf, rgid, rpeers, rwhy = _fast_redundant_flags_labeled(df, title_col, redundant_similarity)
    df["FLAG_RedundantSameAreaYear"] = rf
    df["RedundantGroupID"] = rgid
    df["RedundantPeers"] = rpeers
    df["Reason_Redundant"] = rwhy

    # ---- Potential Ghost (robust) ----
    amt_hi = np.nanpercentile(df["__Amount"].dropna(), ghost_high_amount_percentile) if df["__Amount"].notna().any() else np.nan
    status_vals = (df[status_col].astype(str).str.lower().str.strip() if status_col else pd.Series("", index=df.index))
    accomp = df["__AccompPct"]

    completeish = status_vals.str.contains("complete") | (accomp.fillna(0) >= 99)
    has_start = df[start_col].notna() if start_col else pd.Series(False, index=df.index)
    has_end   = df[end_col].notna()   if end_col   else pd.Series(False, index=df.index)

    dur_days = (df[end_col] - df[start_col]).dt.days if (start_col and end_col) else pd.Series(np.nan, index=df.index)
    very_short = (dur_days < 7)
    high_amt = (df["__Amount"] >= amt_hi) if np.isfinite(amt_hi) else pd.Series(False, index=df.index)
    long_open = ((pd.Timestamp.now() - df[start_col]).dt.days > 730) if start_col else pd.Series(False, index=df.index)
    no_dates = (~has_start & ~has_end)

    # target overrun
    overrun = pd.Series(False, index=df.index)
    if use_target_overrun and target_col:
        tgt = df[target_col]
        past = tgt.notna() & ((pd.Timestamp.now() - tgt).dt.days > int(grace_days))
        not_complete = ~completeish | ~has_end
        overrun = past & not_complete

    ghost_flag = (
        (completeish & ~has_end) |
        (completeish & very_short & high_amt) |
        (long_open & ~has_end) |
        (no_dates & high_amt) |
        (overrun)
    )

    reasons = []
    reasons.append(np.where(completeish & ~has_end, "Status complete/≈100% but no completion date", ""))
    reasons.append(np.where(completeish & very_short & high_amt, "Completion < 7 days with high amount", ""))
    reasons.append(np.where(long_open & ~has_end, ">2 years since start with no completion date", ""))
    reasons.append(np.where(no_dates & high_amt, "No dates recorded despite high amount", ""))
    if use_target_overrun and target_col:
        reasons.append(np.where(overrun, f"Target completion + {grace_days}d grace elapsed; not completed", ""))

    df["FLAG_PotentialGhost"] = ghost_flag.fillna(False)
    df["Reason_PotentialGhost"] = pd.Series(["; ".join([r for r in row if r]) for row in zip(*reasons)], index=df.index)

    # ---- Never-ending (long duration OR recurring multi-year) ----
    dur_ok = df["__DurationDays"].fillna(-1)
    long_dur = dur_ok.ge(never_ending_days)

    recurring = pd.Series(False, index=df.index)
    rec_reason = [""]*len(df)
    if title_col:
        for (_, _), sub in df.groupby("__AreaKey", dropna=False):
            if sub.empty: continue
            for i, r in sub.iterrows():
                if pd.isna(r["__Year"]) or pd.isna(r[title_col]): 
                    continue
                base = _strip_generic_phrases(str(r[title_col]))
                sim = sub[sub[title_col].astype(str).apply(
                    lambda t: rf_fuzz.token_set_ratio(_strip_generic_phrases(t), base) >= int(round(0.70*100))
                )]
                years = set(sim["__Year"].dropna().astype(int).tolist())
                if len(years) >= 3:
                    recurring.loc[i] = True
                    rec_reason[list(df.index).index(i)] = f"Similar titles across {len(years)} years in same area"

    df["FLAG_NeverEnding"] = (long_dur | recurring)
    df["Reason_NeverEnding"] = np.where(long_dur, f"Duration ≥ {never_ending_days} days", "") + \
                               np.where(recurring, "; " + pd.Series(rec_reason, index=df.index), "")

    # ---- Costly (explicit cost rate + unit + labels) ----
    def iqr_bounds(s: pd.Series, k=1.5):
        s2 = s.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if s2.empty: return np.nan, np.nan
        q1, q3 = s2.quantile(0.25), s2.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            std = s2.std()
            iqr = std if std > 0 else 1.0
        return (q1 - k*iqr, q3 + k*iqr)

    use_km = df["__CostPerKm"].notna().sum() >= df["__CostPerSqKm"].notna().sum()
    metric_name = "__CostPerKm" if use_km else "__CostPerSqKm"
    unit = "₱/km" if use_km else "₱/sq-km"

    low_t, high_t = iqr_bounds(df[metric_name], k=cost_iqr_k)
    pct = df[metric_name].rank(pct=True)
    costly = (df[metric_name] < low_t) | (df[metric_name] > high_t) if np.isfinite(low_t) and np.isfinite(high_t) else pd.Series(False, index=df.index)

    df["CostRate"] = df[metric_name]
    df["CostRateUnit"] = unit
    df["CostRatePercentile"] = pct
    df["FLAG_Costly"] = costly.fillna(False)
    df["CostMetricUsed"] = metric_name
    df["CostOutlierLow"], df["CostOutlierHigh"] = low_t, high_t
    df["Reason_Costly"] = np.where(df["FLAG_Costly"],
                                   f"Outlier by IQR in {unit} (low<{low_t:,.0f} / high>{high_t:,.0f})",
                                   "")

    # ---- outputs ----
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

# ----------------- legacy name (compat shim) -----------------
def compute_project_flags(
    df,
    redundant_similarity=0.70,
    ghost_high_amount_percentile=75,
    never_ending_days=730,
    cost_iqr_k=1.5,
    use_target_overrun=True,
    grace_days=60
):
    prep = preprocess_projects(df)
    return compute_project_flags_fast(
        prep["prepped"], prep["colmap"],
        redundant_similarity=redundant_similarity,
        ghost_high_amount_percentile=ghost_high_amount_percentile,
        never_ending_days=never_ending_days,
        cost_iqr_k=cost_iqr_k,
        use_target_overrun=use_target_overrun,
        grace_days=grace_days
    )
