import re
import pandas as pd
import numpy as np

def _find_col(cols, patterns):
    for c in cols:
        if re.search(patterns, str(c), re.I):
            return c
    return None

def compute_contractor_indicators(
    annotated_df: pd.DataFrame,
    concentration_share=0.30,
    min_repeated_flags=3,
    high_cost_percentile=90
) -> dict:
    """
    Inputs: annotated_full from compute_project_flags (already has helper columns)
    Outputs:
      - contractor_table: one row per contractor with metrics and boolean indicators
      - summary: simple counts
    """
    df = annotated_df.copy()
    cols = list(df.columns)

    contractor_col = _find_col(cols, r"(contractor|supplier|winning\s*bidder|provider|vendor)")
    region_col     = _find_col(cols, r"\bregion\b")
    province_col   = _find_col(cols, r"\bprovince\b")
    city_col       = _find_col(cols, r"(city|municipality|muni|lgu)")

    if contractor_col is None:
        return {"contractor_table": pd.DataFrame(), "summary": pd.DataFrame([{"Note":"No contractor column detected"}])}

    # normalize contractor name (light)
    df["__Contractor"] = df[contractor_col].astype(str).str.strip()

    # area-year key is already in "__AreaKey"; year in "__Year"
    # Concentration: contractor's share of projects within area-year clusters
    grp = df.groupby(["__AreaKey","__Year"])
    conc_rows = []
    for (ak, yr), g in grp:
        if len(g)==0: continue
        total = len(g)
        shares = g["__Contractor"].value_counts(normalize=True)
        for name, sh in shares.items():
            conc_rows.append({"__AreaKey": ak, "__Year": yr, "__Contractor": name, "share_area_year": sh, "n_in_area_year": total})
    conc = pd.DataFrame(conc_rows) if conc_rows else pd.DataFrame(columns=["__AreaKey","__Year","__Contractor","share_area_year","n_in_area_year"])

    # Project-level flags by contractor
    flags = df[[
        "__Contractor","FLAG_RedundantSameAreaYear","FLAG_PotentialGhost","FLAG_NeverEnding","FLAG_Costly",
        "__CostPerKm","__CostPerSqKm"
    ]].copy()

    # Contractor aggregates
    agg = flags.groupby("__Contractor").agg(
        n_projects=("FLAG_Costly","size"),
        n_ghost=("FLAG_PotentialGhost","sum"),
        n_never=("FLAG_NeverEnding","sum"),
        n_costly=("FLAG_Costly","sum"),
        n_redundant=("FLAG_RedundantSameAreaYear","sum"),
        mean_cost_per_km=("__CostPerKm","mean"),
        mean_cost_per_sqkm=("__CostPerSqKm","mean")
    ).reset_index()

    # Outlier rates
    agg["rate_costly"] = agg["n_costly"] / agg["n_projects"]
    agg["n_repeated_issues"] = agg["n_ghost"] + agg["n_never"] + agg["n_costly"]

    # High average unit cost vs peers (choose the denser metric)
    metric_counts = {
        "mean_cost_per_km": agg["mean_cost_per_km"].notna().sum(),
        "mean_cost_per_sqkm": agg["mean_cost_per_sqkm"].notna().sum()
    }
    unit_metric = "mean_cost_per_km" if metric_counts["mean_cost_per_km"] >= metric_counts["mean_cost_per_sqkm"] else "mean_cost_per_sqkm"
    pctl_cut = np.nanpercentile(agg[unit_metric], high_cost_percentile) if agg[unit_metric].notna().any() else np.nan
    agg["IND_HighMeanUnitCost"] = (agg[unit_metric] >= pctl_cut) if pd.notna(pctl_cut) else False

    # Concentration indicator: if contractor exceeds share threshold in any area-year
    if not conc.empty:
        conc_flag = conc.groupby("__Contractor")["share_area_year"].max().reset_index().rename(columns={"share_area_year":"max_share_area_year"})
    else:
        conc_flag = pd.DataFrame({"__Contractor": agg["__Contractor"], "max_share_area_year": np.nan})
    agg = agg.merge(conc_flag, on="__Contractor", how="left")
    agg["IND_Concentration"] = agg["max_share_area_year"].fillna(0) >= float(concentration_share)

    # Repeated issues indicator
    agg["IND_RepeatedIssues"] = agg["n_repeated_issues"] >= int(min_repeated_flags)

    # Cost outlier rate (no fixed threshold; report as %; optional decision threshold if needed)
    # Build final contractor table
    contractor_table = agg.rename(columns={
        "__Contractor":"Contractor",
        "n_projects":"Projects",
        "n_ghost":"GhostFlags",
        "n_never":"NeverEndingFlags",
        "n_costly":"CostlyFlags",
        "n_redundant":"RedundantFlags",
        "rate_costly":"RateCostlyOutliers",
        "max_share_area_year":"MaxShareInAreaYear",
    }).sort_values(["IND_RepeatedIssues","IND_Concentration","IND_HighMeanUnitCost","RateCostlyOutliers","Projects"], ascending=[False,False,False,False,False])

    summary = pd.DataFrame({
        "Indicator":[
            "Contractors (detected)",
            "With Concentration flag",
            "With Repeated Issues flag",
            f"High Mean Unit Cost ≥ p{high_cost_percentile}",
        ],
        "Count":[
            contractor_table.shape[0],
            contractor_table["IND_Concentration"].sum(),
            contractor_table["IND_RepeatedIssues"].sum(),
            contractor_table["IND_HighMeanUnitCost"].sum()
        ]
    })

    return {
        "contractor_table": contractor_table,
        "summary": summary
    }
