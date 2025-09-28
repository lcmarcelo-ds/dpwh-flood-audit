# app.py
from pathlib import Path
import textwrap
import pandas as pd
import streamlit as st

from dpwhlib.io import read_base_csv_from_path, save_csv_bytes
from dpwhlib.flags import preprocess_projects, compute_project_flags_fast
from dpwhlib.contractor import compute_contractor_indicators

st.set_page_config(page_title="DPWH Flood-Control Audit", layout="wide")

# ---------- Data path (bundled; no uploads) ----------
DATA_DIR = Path(__file__).parent / "data"
BASE_CSV = DATA_DIR / "Flood_Control_Data.csv"

st.title("DPWH Flood-Control Audit")
st.caption("Rules-based screening using the flood control data; includes contractor indicators.")

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

# ---------- Sidebar: thresholds ----------
st.sidebar.header("Project Flag Thresholds")
redund_sim = st.sidebar.slider(
    "Redundant: title similarity (0–1)", 0.40, 0.95, 0.70, step=0.01,
    help="Same area + same year; requires this similarity AND ≥2 uncommon tokens in common."
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

# ---------- Sidebar: column overrides & extra rules ----------
st.sidebar.header("Columns (optional overrides)")
cols = list(df.columns)
def _sel(label): return st.sidebar.selectbox(label, ["(auto)"] + cols, index=0)

title_col_sel = _sel("Project title")
amount_col_sel = _sel("Amount / Contract cost")
status_col_sel = _sel("Status or % complete")
start_col_sel  = _sel("Start / NTP date")
end_col_sel    = _sel("Completion date (actual)")
target_col_sel = _sel("Target completion date")
year_col_sel   = _sel("Year")
region_col_sel = _sel("Region")
prov_col_sel   = _sel("Province")
city_col_sel   = _sel("City/Municipality")
brgy_col_sel   = _sel("Barangay")
contractor_col_sel = _sel("Contractor/Supplier")
length_col_sel = _sel("Length (m/km)")
area_col_sel   = _sel("Area (sqm/hectares)")

st.sidebar.header("Extra rules")
use_target_overrun = st.sidebar.checkbox(
    "Use target-completion overrun rule", value=True,
    help="If target completion exists, project not completed, and target+grace has passed → flag as potential ghost."
)
grace_days = st.sidebar.number_input(
    "Grace days for target overrun", min_value=0, max_value=365, value=60, step=5,
    help="Additional grace period after target completion before overrun is flagged."
)

col_overrides = {
    k: (None if v == "(auto)" else v) for k, v in dict(
        title=title_col_sel,
        amount=amount_col_sel,
        status=status_col_sel,
        start=start_col_sel,
        end=end_col_sel,
        target=target_col_sel,
        year=year_col_sel,
        region=region_col_sel,
        province=prov_col_sel,
        city=city_col_sel,
        barangay=brgy_col_sel,
        contractor=contractor_col_sel,
        length=length_col_sel,
        area=area_col_sel
    ).items()
}

with st.sidebar.expander(" About these thresholds"):
    st.markdown("""
- **Sensitivity vs specificity:** Lower thresholds flag **more** items; higher thresholds flag **fewer** but stronger signals.  
- **Dataset-relative:** Percentiles and IQR are computed **within your loaded file**.  
- **Transparency:** Thresholds are shown so the public sees exactly how flags were derived.  
- **Not findings:** Flags are **starting points** for doc/site verification.
""")

# ---------- Cache heavy preprocessing ----------
@st.cache_data(show_spinner=False)
def _preprocess_once(df_input: pd.DataFrame, overrides: dict):
    return preprocess_projects(df_input, overrides=overrides)

with st.spinner("Preparing data (one-time)…"):
    prep = _preprocess_once(df, col_overrides)
prepped_df, colmap = prep["prepped"], prep["colmap"]

# Show a tiny detection report
with st.expander("Column detection (for transparency)"):
    st.json(colmap)

# ---------- Compute flags & indicators (fast) ----------
with st.spinner("Computing screening indicators…"):
    proj = compute_project_flags_fast(
        prepped_df, colmap,
        redundant_similarity=redund_sim,
        ghost_high_amount_percentile=ghost_hi_pct,
        never_ending_days=never_days,
        cost_iqr_k=cost_iqr_k,
        use_target_overrun=use_target_overrun,
        grace_days=int(grace_days)
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
    # show reasons/labels to make review faster
    cols_to_show = [
        colmap.get("title") or "Title",
        "FLAG_RedundantSameAreaYear","RedundantGroupID","Reason_Redundant",
        "FLAG_PotentialGhost","Reason_PotentialGhost",
        "FLAG_NeverEnding","Reason_NeverEnding",
        "FLAG_Costly","Reason_Costly",
        "CostRate","CostRateUnit","CostRatePercentile"
    ]
    for key, label in [
        ("all_flagged", "All flagged projects"),
        ("redundant", "Redundant (same area + same year + similar titles)"),
        ("ghost", "Potential ghost (status/date inconsistencies & target overrun)"),
        ("neverending", "Never-ending (long duration or recurring titles)"),
        ("costly", "Costly (₱/km or ₱/sq-km outliers)")
    ]:
        dfv = proj.get(key, pd.DataFrame())
        st.caption(f"{label} — {len(dfv):,} rows")
        show = [c for c in cols_to_show if c in dfv.columns]
        st.dataframe((dfv[show] if show else dfv).head(150), use_container_width=True)

with t2:
    st.subheader("Contractor Indicators")
    st.markdown("""
These are **screening indicators** to prioritize review. They do **not** prove wrongdoing. 
Use them to queue **document checks** (POW, plans/estimates, inspection, completion/acceptance) 
and **site verification** before any conclusion.
    """)
    with st.expander("🔎 What the indicators mean"):
        st.markdown("""
**a) Concentration (Share within an area–year)** → competition risk screen.  
**b) Repeated Issues** → count of ghost/never-ending/costly flags per contractor.  
**c) Cost Outlier Rate** → fraction of contractor projects flagged as costly.  
**d) High Mean Unit Cost** → contractor’s mean ₱/km or ₱/sq-km ≥ chosen peer percentile.
        """)
    st.write("**Contractor Summary Table**")
    st.dataframe(contr["contractor_table"].head(150), use_container_width=True)

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
- **Redundant:** Same area & year; similarity ≥ **{redund_sim:.2f}** and ≥2 uncommon tokens in common.  
- **Potential “Ghost”:** status/date logic; “high-amount” = top **{ghost_hi_pct}th** percentile; optional **target overrun** with {grace_days}-day grace.  
- **Never-ending:** duration ≥ **{never_days}** days or recurring similar titles across ≥ 3 years in the same area.  
- **Costly:** **IQR** outliers on ₱/km or ₱/sq-km (k = **{cost_iqr_k}**), with explicit **CostRate** and unit.
    """)
