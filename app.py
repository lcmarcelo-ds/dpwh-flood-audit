import streamlit as st
from pathlib import Path
import pandas as pd

from dpwhlib.io import read_base_csv_from_path, save_csv_bytes
from dpwhlib.flags import compute_project_flags
from dpwhlib.contractor import compute_contractor_indicators

st.set_page_config(page_title="DPWH Flood-Control Screening (Projects + Contractors)", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
BASE_CSV = DATA_DIR / "Flood_Control_Data.csv"

st.title("DPWH Flood-Control Screening (Data-Only)")
st.caption("Rules-based screening using the bundled dataset; adds contractor indicators. No uploads required.")

# --- Explanations (public) ---
with st.expander("How to read this dashboard"):
    st.markdown("""
**What this is:** A *screening* tool to prioritize verification.  
**What it uses:** Only the dataset bundled under `/data/Flood_Control_Data.csv`.  
**What it shows:** Project-level flags (redundant, ghost, never-ending, costly) **and** contractor indicators (concentration, repeated issues, outlier rates).  
**Not legal findings:** Always verify with records and site inspection.
""")

# Load
if not BASE_CSV.exists():
    st.error("Missing /data/Flood_Control_Data.csv. Please place the file in the data folder.")
    st.stop()

df = read_base_csv_from_path(BASE_CSV)

# Tunables (for transparency)
st.sidebar.header("Flag Thresholds (tune for analysis)")
ghost_hi_pct = st.sidebar.slider("High-amount threshold (percentile)", 50, 95, 75, step=1)
never_days = st.sidebar.number_input("Never-ending: min days", min_value=365, max_value=1825, value=730, step=15)
redund_sim = st.sidebar.slider("Redundant title similarity (0–1)", 0.4, 0.95, 0.60, step=0.01)
cost_iqr_k = st.sidebar.slider("Cost outlier IQR multiplier", 0.5, 3.0, 1.5, step=0.1)

st.sidebar.header("Contractor Thresholds")
contr_share = st.sidebar.slider("Concentration: share within area-year", 0.1, 0.9, 0.30, step=0.05)
contr_repeat = st.sidebar.number_input("Repeated issues: min flagged projects", min_value=1, max_value=50, value=3)
contr_cost_pctl = st.sidebar.slider("High avg cost per km/sqkm: percentile", 60, 99, 90, step=1)

# Compute project flags
proj = compute_project_flags(
    df,
    redundant_similarity=redund_sim,
    ghost_high_amount_percentile=ghost_hi_pct,
    never_ending_days=never_days,
    cost_iqr_k=cost_iqr_k
)

# Contractor indicators
contr = compute_contractor_indicators(
    proj["annotated_full"],
    concentration_share=contr_share,
    min_repeated_flags=contr_repeat,
    high_cost_percentile=contr_cost_pctl
)

# Summary
st.header("Summary")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Projects")
    st.dataframe(proj["summary"], use_container_width=True)
with c2:
    st.subheader("Contractors")
    st.dataframe(contr["summary"], use_container_width=True)

# Tabs
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
        ("redundant", "Redundant (same area + year + similar titles)"),
        ("ghost", "Potential ghost"),
        ("neverending", "Never-ending"),
        ("costly", "Costly (₱/km or ₱/sq-km outliers)")
    ]:
        dfv = proj.get(key, pd.DataFrame())
        st.caption(f"{label} — {len(dfv):,} rows")
        st.dataframe(dfv.head(100), use_container_width=True)

with t2:
    st.subheader("Contractor Indicators")
    st.markdown("""
**What we compute (data-only):**
- **Concentration**: contractor's project share within the same *area–year* (proxy for market dominance patterns).  
- **Repeated Issues**: contractor has ≥ N projects flagged (ghost / never-ending / costly).  
- **Cost Outlier Rate**: fraction of contractor’s projects flagged as costly outliers.  
- **High Average Cost**: contractor’s mean ₱/km or ₱/sq-km > chosen percentile vs. peers.  
    """)
    st.write("**Contractor Summary Table**")
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
        ("contractor_indicators.csv", contr["contractor_table"])
    ]:
        st.download_button(f"Download {label}", data=save_csv_bytes(dfv), file_name=label, mime="text/csv")

with t4:
    st.subheader("Legal Basis (PH)")
    st.markdown("""
- **Procurement / Implementation:** Updated IRR of **RA 9184** (as of 19 Jul 2024) and **RA 12009 (New GPRA)** govern planning → bidding → **contract implementation** (inspection, completion, acceptance).  
- **Audit Authority:** **PD 1445** (Government Auditing Code) mandates COA’s post-audit and inspection.  
- **Documentation:** **COA** circulars and audit guides emphasize **completion evidence, inspection reports, acceptance**, and documentary sufficiency.
    """)
    st.subheader("Data-Science Basis")
    st.markdown(f"""
- **Redundant:** Title similarity ≥ **{redund_sim:.2f}** within same area & year.  
- **Potential “Ghost”:** logic on status/dates; “high-amount” = top **{ghost_hi_pct}th** percentile in this dataset.  
- **Never-ending:** Duration ≥ **{never_days}** days or recurring similar titles across ≥ 3 years in same area.  
- **Costly:** **IQR** outliers on ₱/km or ₱/sq-km with k = **{cost_iqr_k}**.  
- **Contractor:** data-only rates (concentration, repeated flags, outlier rates, high mean unit cost vs. peers).
    """)
    st.subheader("Policy Approach")
    st.markdown("""
- **Targeted Verification:** Flags prioritize file review (POW, plans/estimates, inspection, completion/acceptance) and site checks.  
- **Transparency & Due Process:** Indicators are not findings; escalate only after validation.  
- **Data Standards:** Normalize locations (PSGC) and contractor names internally to improve traceability (future step).
    """)
