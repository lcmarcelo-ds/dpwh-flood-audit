
from pathlib import Path
import textwrap
import pandas as pd
import streamlit as st

from dpwhlib.io import read_base_csv_from_path, save_csv_bytes
from dpwhlib.flags import preprocess_projects, compute_project_flags_fast
from dpwhlib.contractor import compute_contractor_indicators

st.set_page_config(page_title="DPWH Flood-Control Screening (Projects + Contractors)", layout="wide")

# ---------- Data path (bundled; no uploads) ----------
DATA_DIR = Path(__file__).parent / "data"
BASE_CSV = DATA_DIR / "Flood_Control_Data.csv"

st.title("DPWH Flood-Control Audit ")
st.caption("Rules-based screening using the bundled dataset; includes contractor indicators. No uploads required.")

with st.expander("How to read this dashboard"):
    st.markdown("""
**What this is:** A *screening* tool to prioritize verification.  
**What it uses:** Flood Control Data from DPWH Website and Sumbongpangulo.ph.  
**What it shows:** Project-level flags (redundant, ghost, never-ending, costly) **and** contractor indicators (concentration, repeated issues, outlier rates).  
**Not legal findings:** Always verify with records and site inspection before conclusions.
""")

# ---------- Robust load guard ----------
if not BASE_CSV.exists():
    st.error("Missing `/data/Flood_Control_Data.csv`. Please add the file and rerun.")
    st.stop()

try:
    df = read_base_csv_from_path(BASE_CSV)
except Exception as e:
    st.error(f"Could not read `/data/Flood_Control_Data.csv`\n\nError: {e}")
    raw = BASE_CSV.read_bytes()
    head = raw[:2048].decode("utf-8", errors="ignore")
    st.code(textwrap.shorten(head, width=2000, placeholder="…"), language="text")
    st.stop()

if df.empty or df.columns.size == 0:
    st.error("The file was read but contains no columns/rows. Check delimiter and encoding (CSV may be ';' or tab).")
    st.stop()

# ---------- Sidebar thresholds + explanations ----------
st.sidebar.header("Flag Thresholds")
redund_sim = st.sidebar.slider(
    "Redundant: title similarity (0–1)", 0.40, 0.95, 0.60, step=0.01,
    help="Projects in the SAME area & year are marked redundant if titles are ≥ this similarity. "
         "Lower = more sensitive; higher = stricter."
)
ghost_hi_pct = st.sidebar.slider(
    "Potential Ghost: 'high-amount' percentile", 50, 95, 75, step=1,
    help="Defines 'high-amount' by percentile within this dataset; used by ghost logic."
)
never_days = st.sidebar.number_input(
    "Never-ending: minimum duration (days)", min_value=365, max_value=1825, value=730, step=15,
    help="Projects with duration ≥ this are flagged. If no end date, uses today − start."
)
cost_iqr_k = st.sidebar.slider(
    "Costly: IQR multiplier (k)", 0.5, 3.0, 1.5, step=0.1,
    help="Outlier threshold for ₱/km or ₱/sq-km via IQR. Lower k = more sensitive."
)

st.sidebar.header("Contractor Thresholds")
contr_share = st.sidebar.slider(
    "Concentration: max share in any area–year", 0.10, 0.90, 0.30, step=0.05,
    help="Flag if contractor's share of projects in any area–year cluster ≥ this value."
)
contr_repeat = st.sidebar.number_input(
    "Repeated issues: minimum flagged projects", min_value=1, max_value=100, value=3, step=1,
    help="Flag contractor if (ghost + never-ending + costly) flags ≥ this number."
)
contr_cost_pctl = st.sidebar.slider(
    "High mean unit cost: peer percentile", 60, 99, 90, step=1,
    help="Flag if contractor mean ₱/km (or ₱/sq-km) ≥ this percentile vs. peers."
)

with st.sidebar.expander(" About these thresholds"):
    st.markdown("""
- **Sensitivity vs specificity:** Lower thresholds flag **more** items; higher thresholds flag **fewer** but stronger signals.  
- **Dataset-relative:** Percentiles and IQR are computed **within your loaded file**.  
- **Transparency:** Thresholds are shown so the public sees exactly how flags were derived.  
- **Not findings:** Flags are **starting points** for doc/site verification.
""")

# ---------- Cache heavy preprocessing ----------
@st.cache_data(show_spinner=False)
def _preprocess_once(df_input: pd.DataFrame):
    return preprocess_projects(df_input)

with st.spinner("Preparing data (one-time)…"):
    prep = _preprocess_once(df)
prepped_df, colmap = prep["prepped"], prep["colmap"]

# ---------- Compute flags & indicators (fast) ----------
with st.spinner("Computing screening indicators…"):
    proj = compute_project_flags_fast(
        prepped_df, colmap,
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
These are **screening indicators** to prioritize review. They do **not** prove wrongdoing. 
Use them to queue **document checks** (POW, plans/estimates, inspection, completion/acceptance) 
and **site verification** before any conclusion.
    """)

    with st.expander("🔎 What the indicators mean"):
        st.markdown("""
**a) Concentration (Share within an area–year)**  
We compute, for each contractor, their **share of all projects** within the same area (Region/Province/City) and **year**.  
A **high share** (e.g., ≥ 30%) may signal **reduced competition** → check bid histories (e.g., single-bidder cases).  
*Legal anchor:* Procurement must be **competitive & transparent** under **RA 9184 / RA 12009 IRR**.
        """)
        st.markdown("""
**b) Repeated Issues (sum of flagged projects)**  
Total of a contractor’s **Potential Ghost**, **Never-ending**, or **Costly** flags.  
If the total crosses a threshold (e.g., **≥ 3**), it’s a **risk signal** for performance review and closer audit scrutiny.  
*Legal anchor:* **PD 1445** (COA mandate) and **RA 9184** (sanctions/blacklisting) **after due process**.
        """)
        st.markdown("""
**c) Cost Outlier Rate**  
The **fraction** of a contractor’s projects flagged as **cost outliers** via **IQR**.  
Persistent high rates suggest price reasonableness review (compare POW/estimates vs. outcomes).
        """)
        st.markdown("""
**d) High Mean Unit Cost (peer comparison)**  
Contractor’s mean ₱/km (or ₱/sq-km) compared with peers; ≥ chosen percentile (e.g., **90th**) is flagged.  
Aligns with **Value for Money** principle in procurement; check terrain/scope context.
        """)

    with st.expander(" How the numbers are computed"):
        st.markdown(f"""
- **Concentration:** For each *area–year*, contractor share = (projects by contractor) ÷ (total projects).  
  Flag if **max share** across area–years ≥ sidebar threshold (default **{contr_share:.0%}**).  
- **Repeated Issues:** sum of **ghost + never-ending + costly**; flag if ≥ sidebar threshold (default **{contr_repeat}**).  
- **Cost Outlier Rate:** (# **costly**) ÷ (total).  
- **High Mean Unit Cost:** uses the **denser** metric (₱/km if length is more complete, else ₱/sq-km).  
  Flag if contractor mean ≥ peer percentile (default **p{contr_cost_pctl}**).  
- **Robust:** Uses **median/IQR**; thresholds are **transparent & tunable**.
        """)

    with st.expander(" Due process & safeguards"):
        st.markdown("""
- **Not a finding:** Indicators are **triage**. Any sanction requires **records + site validation** and **due process**.  
- **Context matters:** High concentration can reflect **few qualified bidders**; high costs can reflect **difficult sites**.  
- **Documentation:** Verify completion/acceptance, inspection reports, **ORS/BURS**, etc.  
- **Data standards:** Adopt **PSGC** naming in future to improve grouping.
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
        ("contractor_indicators.csv", contr["contractor_table"]),
    ]:
        st.download_button(f"Download {label}", data=save_csv_bytes(dfv), file_name=label, mime="text/csv")

with t4:
    st.subheader("Legal • Data-Science • Policy ")
    st.markdown("""
- **Procurement & Contract Implementation:** Updated IRR of **RA 9184** and **RA 12009 (New GPRA)** cover planning → bidding → **contract implementation** (inspection, completion, acceptance).
- **Audit Authority:** **PD 1445** mandates **COA** examination of records and post-audit/inspections.
- **Documentation:** COA circulars and procurement audit guides emphasize **completion evidence**, inspection reports, and acceptance documents.
    """)
    st.markdown(f"""
**Data-Science Basis**  
- **Redundant:** Title similarity ≥ **{redund_sim:.2f}** within same area & year.  
- **Potential “Ghost”:** status/date logic; “high-amount” = top **{ghost_hi_pct}th** percentile of this dataset.  
- **Never-ending:** duration ≥ **{never_days}** days or recurring similar titles across ≥ 3 years in the same area.  
- **Costly:** **IQR** outliers on ₱/km or ₱/sq-km (k = **{cost_iqr_k}**).  
- **Contractor:** concentration, repeated flags, outlier rates, and high mean unit cost are **data-only indicators**.
    """)
