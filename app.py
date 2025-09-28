import pandas as pd
import streamlit as st
from pathlib import Path

from dpwhlib.pipeline import compute_all_flags
from dpwhlib.io import read_base_csv_from_path, read_budget_xlsx_from_path, save_csv_bytes
from dpwhlib.matching import run_all_strategies_with_templates

st.set_page_config(page_title="DPWH Flood-Control Screening (Public)", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
BASE_CSV = DATA_DIR / "Flood_Control_Data.csv"
BUDGET_XLSX = DATA_DIR / "DPWH-budget.xlsx"

st.title("DPWH Flood-Control Audit ")


# ---------- Auto-explanation (public-facing) ----------
st.header("Auto-Explanation")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Legal Basis (PH)")
    st.markdown("""
- **Procurement rules.** Updated IRR of **RA 9184** (19 Jul 2024) and **RA 12009** (New Government Procurement Act) govern planning → bidding → **contract implementation** (inspection, completion, acceptance).  
- **Audit authority.** **PD 1445** (Government Auditing Code) empowers COA to examine records and conduct post-audit/inspection.  
- **Documentary requirements.** **COA Circular 2023-004** lists docs for common transactions This Circular updates only the documentary requirements for disbursements
relating to the following common government transactions chargeable against the
funds of all NGAs, GCs and LGUs:
a) Cash Advances;
b) Fund Transfers to Non-Government Organizations/ People's
Organizations/Civil Society Organizations (NGOs/POs/CSOs);
c) Fund Transfers to Implementing Agencies;
d) Salary;
e) Allowances, Honoraria and Other Forms ofCompensation;
f) Other Expenditures;
g) Extraordinary and Miscellaneous Expenses;
h) Prisoner's Subsistence Allowance;
i) Procurement of Goods, Consultancy and Infrastructure Projects
(Regardless ofMode of Procurement);
j) Cultural and Athletic Activities;
k) Human Resource Development and Training Program;
1) Financial Expenses;
m) Legal Retainer's Fee; and
n) Road Right-of-Way (ROW)/ Real Property.**.
    """)

with col2:
    st.subheader("Data-Science Basis")
    st.markdown("""
We compute **four indicators**:
1) **Redundant**: same area + same year + highly similar project titles.  
2) **Potential “ghost”**: status says completed but **no completion date**, or **very short duration** for high-amount items, or **no dates** on high-amount items, or **>2 years** ongoing with no completion.  
3) **Never-ending**: duration ≥ **730 days** or highly similar titles recur in the same area across **≥3 years**.  
4) **Costly**: **₱/km** or **₱/sq-km** outliers via robust IQR.
> These are **screening heuristics** to prioritize document and site verification; not findings.
    """)

with col3:
    st.subheader("Policy Approach")
    st.markdown("""
- **Targeted verification**: Use flags to queue project files (POW, plans/estimates, CPES, tests, **completion/acceptance**) and **site checks**.  
- **Data standards**: Normalize locations with ** Philippine Standard Geographic Code (PSGC) **; standardize titles/codes to improve traceability across agencies.  
- **Transparency**: Publish aggregates and anonymized flags; escalate only after verification and due process.
    """)

st.divider()

# ---------- Load bundled data ----------
if not BASE_CSV.exists() or not BUDGET_XLSX.exists():
    st.error("Missing data files in /data. Please place Flood_Control_Data.csv and DPWH-budget.xlsx there.")
    st.stop()

base_df = read_base_csv_from_path(BASE_CSV)
budget_df, used_sheet = read_budget_xlsx_from_path(BUDGET_XLSX)

# ---------- Compute flags on load ----------
with st.spinner("Computing screening indicators…"):
    outs = compute_all_flags(base_df)

st.success("Screening complete.")

# Summary
st.subheader("Flag Summary")
st.dataframe(outs["summary"], use_container_width=True)

# Tabbed views
t1, t2, t3, t4, t5 = st.tabs(["All Flagged", "Redundant", "Potential Ghost", "Never-ending", "Costly"])
for tab, key, label in [
    (t1, "all_flagged", "All Flagged"),
    (t2, "redundant", "Redundant"),
    (t3, "ghost", "Potential Ghost"),
    (t4, "neverending", "Never-ending"),
    (t5, "costly", "Costly"),
]:
    with tab:
        df = outs[key]
        st.caption(f"{label} — {len(df):,} rows")
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button(f"Download {label} CSV", data=save_csv_bytes(df), file_name=f"{key}.csv", mime="text/csv")

st.divider()

# ---------- Budget linkage (review-only; no auto-merge) ----------
st.header("Budget Linkage (review-first, no auto-merge)")
st.caption(f"Budget workbook sheet auto-selected: **{used_sheet}**")

with st.spinner("Preparing deterministic and fuzzy candidate links (review-only)…"):
    used_sheet_name, report_df, files_dict = run_all_strategies_with_templates(
        base_df=base_df,
        budget_df=budget_df,
        psgc_map_bytes=None,     # public demo uses raw data; add maps in repo if needed
        title_map_bytes=None,
        code_map_bytes=None
    )

st.dataframe(report_df, use_container_width=True)

# Offer downloads for templates / candidate lists present
for label, df in files_dict.items():
    if isinstance(df, pd.DataFrame):
        st.write(f"**{label}**  — {len(df):,} rows")
        st.dataframe(df.head(50), use_container_width=True)
        st.download_button(f"Download {label}", data=save_csv_bytes(df), file_name=label, mime="text/csv")

st.divider()
st.caption("Disclaimer: This public demo provides screening indicators only. Validate with records and site verification before any conclusions.")
