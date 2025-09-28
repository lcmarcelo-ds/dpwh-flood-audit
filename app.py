# app.py
from pathlib import Path
import textwrap
import pandas as pd
import streamlit as st

from dpwhlib.io import read_base_csv_from_path, save_csv_bytes
from dpwhlib.flags import compute_project_flags
from dpwhlib.contractor import compute_contractor_indicators

st.set_page_config(page_title="DPWH Flood-Control Screening (Projects + Contractors)", layout="wide")

# ---------- Data path (bundled; no uploads) ----------
DATA_DIR = Path(__file__).parent / "data"
BASE_CSV = DATA_DIR / "Flood_Control_Data.csv"

st.title("DPWH Flood-Control Audit")
st.caption("Rules-based screening using the bundled dataset; includes contractor indicators. No uploads required.")

with st.expander("How to read this dashboard"):
    st.markdown("""
**What this is:** A *screening* tool to prioritize verification.  
**What it uses:** Flood Control from DPWH and Sumbongsapangulo.ph.  
**What it shows:** Project-level flags (redundant, ghost, never-ending, costly) **and** contractor indicators (concentration, repeated issues, outlier rates).  
**Not legal findings:** Always verify with records and site inspection before conclusions.
""")

# ---------- Robust load guard (handles delimiter/encoding quirks) ----------
if not BASE_CSV.exists():
    st.error("Missing `/data/Flood_Control_Data.csv`. Please add the file and rerun.")
    st.stop()

try:
    df = read_base_csv_from_path(BASE_CSV)
except Exception as e:
    st.error(f"Could not read `/data/Flood_Control_Data.csv`\n\nError: {e}")
    # Show first bytes to aid diagnosis
    raw = BASE_CSV.read_bytes()
    head = raw[:2048].decode("utf-8", errors="ignore")
    st.code(textwrap.shorten(head, width=2000, placeholder="…"), language="text")
    st.stop()

if df.empty or df.columns.size == 0:
    st.error("The file was read but contains no columns/rows. Check delimiter and encoding (CSV may be ';' or tab).")
    st.stop()

# ---------- Sidebar: transparent, tunable thresholds ----------
st.sidebar.header("Project Flag Thresholds")
redund_sim = st.sidebar.slider("Redundant: title similarity (0–1)", 0.40, 0.95, 0.60, step=0.01)
ghost_hi_pct = st.sidebar.slider("Potential Ghost: 'high-amount' percentile", 50, 95, 75, step=1)
never_days = st.sidebar.number_input("Never-ending: minimum days", min_value=365, max_value=1825, value=730, step=15)
cost_iqr_k = st.sidebar.slider("Costly: IQR multiplier (k)", 0.5, 3.0, 1.5, step=0.1)

st.sidebar.header("Contractor Thresholds")
contr_share = st.sidebar.slider("Concentration: max share in any area–year", 0.10, 0.90, 0.30, step=0.05)
contr_repeat = st.sidebar.number_input("Repeated issues: min flagged projects", min_value=1, max_value=100, value=3, step=1)
contr_cost_pctl = st.sidebar.slider("High mean unit cost: percentile", 60, 99, 90, step=1)

# ---------- Compute flags & indicators ----------
with st.spinner("Computing screening indicators…"):
    proj = compute_project_flags(
        df,
        redundant_similarity=redund_sim,
        ghost_high_amount_percentile=ghost_hi_pct,
        never_ending_days=never_days,
        cost_iqr_k=cost_iqr_k
    )
    contr = compute_contractor_indicators(
        proj["annotated_full"],
        concentration_share=contr_share,
        min_repeated_flags=contr_repeat,
        high_cost_percentile=contr_cost_pctl
    )
st.success("Done.")

# ---------- Summary ----------
st.header("Summary")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Projects")
    st.dataframe(proj["summary"], use_container_width=True)
with c2:
    st.subheader("Contractors")
    st.dataframe(contr["summary"], use_container_width=True)

# ---------- Tabs ----------
t1, t2, t3, t4 = st.tabs([
    "Project Flags",
    "Contractor Indicators",
    "Downloads",
    "Legal • Data-Science • Policy"
])

with t1:
    st.subheader("Project Flags")
    for key, label in [
        ("all_flagged", "All flagged projects"),
        ("redundant", "Redundant (same area + same year + similar titles)"),
        ("ghost", "Potential ghost (status/date inconsistencies)"),
        ("neverending", "Never-ending (long duration or recurring titles)"),
        ("costly", "Costly (₱/km or ₱/sq-km outliers)")
    ]:
        dfv = proj.get(key, pd.DataFrame())
        st.caption(f"{label} — {len(dfv):,} rows")
        st.dataframe(dfv.head(100), use_container_width=True)

with t2:
    st.subheader("Contractor Indicators")
    st.markdown("""
**Computed from project flags (data-only):**
- **Concentration**: contractor’s max share within any *area–year* cluster.  
- **Repeated Issues**: total of ghost/never-ending/costly flags across the contractor’s projects ≥ threshold.  
- **Cost Outlier Rate**: fraction of projects flagged as costly outliers.  
- **High Mean Unit Cost**: contractor’s mean ₱/km (or ₱/sq-km) ≥ chosen percentile vs. peers.
    """)
    st.dataframe(contr["contractor_table"].head(100), use_container_width=True)

with t3:
    st.subheader("Download CSVs")
    for label, dfv in [
        ("annotated_full.csv", proj["annotated_full"]),
        ("redundant.csv", proj["redundant"]),
        ("potential_ghost.csv", proj["ghost"]),
        ("neverending.csv", proj["neverending"]),
        ("costly.csv", proj["costly"]),
        ("all_flagged.csv", proj["all_flagged"]),
        ("contractor_indicators.csv", contr["contractor_table"]),
    ]:
        st.download_button(f"Download {label}", data=save_csv_bytes(dfv), file_name=label, mime="text/csv")

with t4:
    st.subheader("Legal Basis (Philippines)")
    st.markdown("""
- **Procurement & Contract Implementation:** Updated IRR of **RA 9184** (19 Jul 2024) and **RA 12009 (New GPRA)** frame planning → bidding → **contract implementation** (inspection, completion, acceptance).
- **Audit Authority:** **PD 1445** (Government Auditing Code) mandates COA’s examination of records and post-audit/inspections.
- **Documentation:** COA circulars and procurement audit guides emphasize **completion evidence**, inspection reports, and acceptance documents.
    """)
    st.subheader("Data-Science Basis")
    st.markdown(f"""
- **Redundant:** Title similarity ≥ **{redund_sim:.2f}** within the same area & year.  
- **Potential “Ghost”:** status/date logic; “high-amount” = top **{ghost_hi_pct}th** percentile of this dataset.  
- **Never-ending:** duration ≥ **{never_days}** days **or** recurring similar titles across ≥ 3 years in the same area.  
- **Costly:** **IQR** outliers on ₱/km or ₱/sq-km (k = **{cost_iqr_k}**).  
- **Contractor:** concentration, repeated flags, outlier rates, and high mean unit cost are **data-only indicators**.
    """)
    st.subheader("Policy Approach")
    st.markdown("""
- **Targeted Verification:** Use indicators to prioritize file review (POW, plans/estimates, CPES, test results, inspection, completion/acceptance) and site checks.  
- **Transparency & Due Process:** Indicators are **not** findings; escalate only after validation.  
- **Data Standards:** Normalize locations (PSGC) and contractor names internally to improve traceability in future iterations.
    """)
