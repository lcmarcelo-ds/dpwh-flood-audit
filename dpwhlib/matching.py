import re
import pandas as pd
from .textutils import norm_text, token_set, seq_ratio, partial_ratio, jaccard, prefix_overlap

def _find_col(df, patterns):
    for c in df.columns:
        if re.search(patterns, str(c), re.I):
            return c
    return None

def _extract_year(v):
    import numpy as np, re
    s = str(v); m = re.search(r"(20\d{2}|19\d{2})", s)
    return int(m.group(1)) if m else np.nan

def run_all_strategies_with_templates(base_df: pd.DataFrame, budget_df: pd.DataFrame,
                                      psgc_map_bytes=None, title_map_bytes=None, code_map_bytes=None):
    files = {}

    # Templates to help canonicalization (for public publishing)
    psgc_cols = [
        ("Region", _find_col(base_df, r"\bregion\b"), _find_col(budget_df, r"\bregion\b")),
        ("Province", _find_col(base_df, r"\bprovince\b"), _find_col(budget_df, r"\bprovince\b")),
        ("CityMunicipality", _find_col(base_df, r"(city|municipality|muni|lgu)"), _find_col(budget_df, r"(city|municipality|muni|lgu)")),
    ]
    psgc_rows = []
    for label, bc, gc in psgc_cols:
        left_vals = sorted({str(v).strip() for v in (base_df[bc] if bc else []) if pd.notna(v)})[:2000]
        right_vals = sorted({str(v).strip() for v in (budget_df[gc] if gc else []) if pd.notna(v)})[:2000]
        n = max(len(left_vals), len(right_vals), 1)
        left_vals += [""]*(n-len(left_vals)); right_vals += [""]*(n-len(right_vals))
        psgc_rows.append(pd.DataFrame({
            f"{label}_Base": left_vals, f"{label}_Budget": right_vals,
            f"{label}_Canonical": [""]*n, "Field": [label]*n
        }))
    files["PSGC_Map_Template.csv"] = pd.concat(psgc_rows, ignore_index=True) if psgc_rows else pd.DataFrame()

    b_title = _find_col(base_df, r"(project|title|name|description)")
    g_title = _find_col(budget_df, r"(project|title|name|description)")
    base_titles = sorted({str(v) for v in base_df[b_title].dropna().unique()})[:5000] if b_title else []
    budg_titles = sorted({str(v) for v in budget_df[g_title].dropna().unique()})[:5000] if g_title else []
    n = max(len(base_titles), len(budg_titles), 1)
    base_titles += [""]*(n-len(base_titles)); budg_titles += [""]*(n-len(budg_titles))
    files["Mapping_TitlePairs_Template.csv"] = pd.DataFrame({
        "Base_Title": base_titles, "Budget_Title": budg_titles, "Title_Canonical": [""]*n
    })

    code_cands_base = [c for c in base_df.columns if re.search(r"(code|id|contract|uacs|pap)", str(c), re.I)]
    code_cands_budget = [c for c in budget_df.columns if re.search(r"(code|id|contract|uacs|pap)", str(c), re.I)]
    files["Mapping_CodePairs_Template.csv"] = pd.DataFrame({
        "Base_Column": [], "Base_Value": [], "Budget_Column": [], "Budget_Value": [], "Canonical_Code": []
    })
    # (Keep templates empty in public demo; fill and re-run in a private workflow if needed.)

    # Fuzzy review list (composite ≥ 0.60) with relaxed blocks
    def prep(df, title_pat, region_pat, prov_pat, city_pat, year_pat):
        t = _find_col(df, title_pat); r = _find_col(df, region_pat)
        p = _find_col(df, prov_pat);  c = _find_col(df, city_pat)
        y = _find_col(df, year_pat)
        out = df.copy()
        out["_title_norm"] = out[t].astype(str).map(norm_text) if t else ""
        out["_region_norm"] = out[r].astype(str).map(norm_text) if r else ""
        out["_prov_norm"] = out[p].astype(str).map(norm_text) if p else ""
        out["_city_norm"] = out[c].astype(str).map(norm_text) if c else ""
        out["_year"] = out[y].apply(_extract_year).astype("Int64") if y else None
        return out, t, r, p, c, y

    base, b_t, b_r, b_p, b_c, b_y = prep(base_df, r"(project|title|name|description)", r"\bregion\b", r"\bprovince\b", r"(city|municipality|muni|lgu)", r"\byear\b")
    budg, g_t, g_r, g_p, g_c, g_y = prep(budget_df, r"(project|title|name|description)", r"\bregion\b", r"\bprovince\b", r"(city|municipality|muni|lgu)", r"\byear\b")

    def block_keys(df):
        keys = []
        for _, r in df.iterrows():
            if r.get("_region_norm") and pd.notna(r.get("_year")):
                keys.append(("region_year", r["_region_norm"], int(r["_year"])))
            elif r.get("_region_norm"):
                keys.append(("region", r["_region_norm"], 0))
            elif r.get("_prov_norm") and pd.notna(r.get("_year")):
                keys.append(("prov_year", r["_prov_norm"], int(r["_year"])))
            elif r.get("_prov_norm"):
                keys.append(("prov", r["_prov_norm"], 0))
            elif pd.notna(r.get("_year")):
                keys.append(("year","",int(r["_year"])))
            else:
                keys.append(("global","",0))
        return keys

    base["_block"] = block_keys(base); budg["_block"] = block_keys(budg)

    SIM = 0.60; TOPK = 5
    num_like = [c for c in budget_df.columns if re.search(r"(budget|amount|appropriation|gaa|nep|gab|allotment|obligation|cost)", str(c), re.I)][:10]
    cands = []
    for blk in sorted(set(base["_block"]).intersection(set(budg["_block"]))):
        b1 = base[base["_block"]==blk]; b2 = budg[budg["_block"]==blk]
        for i, r1 in b1.iterrows():
            t1 = str(r1[b_t]) if b_t else ""
            if not t1: continue
            ts1 = token_set(t1)
            scores = []
            for j, r2 in b2.iterrows():
                t2 = str(r2[g_t]) if g_t else ""
                if not t2: continue
                ts2 = token_set(t2)
                s_seq = seq_ratio(t1, t2)
                s_jac = jaccard(ts1, ts2)
                s_prt = partial_ratio(t1, t2)
                s_pre = prefix_overlap(t1, t2)
                comp = 0.35*s_jac + 0.35*s_prt + 0.20*s_seq + 0.10*s_pre
                if comp >= SIM:
                    row = {
                        "base_index": i, "budget_index": j,
                        "block": f"{blk[0]}|{blk[1]}|{blk[2]}",
                        "sim_composite": round(comp,4),
                        "sim_jaccard": round(s_jac,4),
                        "sim_partial": round(s_prt,4),
                        "sim_seqratio": round(s_seq,4),
                        "sim_prefix": round(s_pre,4),
                        "base_title": t1, "budget_title": t2,
                        "base_region": str(r1.get(b_r,"")), "budget_region": str(r2.get(g_r,"")),
                        "base_province": str(r1.get(b_p,"")), "budget_province": str(r2.get(g_p,"")),
                        "base_city": str(r1.get(b_c,"")), "budget_city": str(r2.get(g_c,"")),
                        "base_year": int(r1["_year"]) if pd.notna(r1["_year"]) else "",
                        "budget_year": int(r2["_year"]) if pd.notna(r2["_year"]) else "",
                    }
                    for c in num_like: row[f"budget::{c}"] = r2.get(c)
                    scores.append(row)
            scores = sorted(scores, key=lambda x: x["sim_composite"], reverse=True)[:TOPK]
            cands.extend(scores)

    files["candidates_fuzzy_flexible.csv"] = pd.DataFrame(cands)

    report = pd.DataFrame({
        "Strategy": ["Fuzzy Flexible (≥0.60)","Templates Generated"],
        "MatchesFound": [len(files["candidates_fuzzy_flexible.csv"]), "PSGC/Title/Code templates ready"],
        "Output": ["candidates_fuzzy_flexible.csv", "PSGC_Map_Template.csv; Mapping_TitlePairs_Template.csv; Mapping_CodePairs_Template.csv"]
    })
    return ("<budget-sheet>", report, files)
