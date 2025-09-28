# app.py
from pathlib import Path
import textwrap
import pandas as pd
import streamlit as st

from dpwhlib.io import read_base_csv_from_path, save_csv_bytes
from dpwhlib.flags import preprocess_projects, compute_project_flags_fast, __FLAGSLIB_VERSION__
from dpwhlib.contractor import compute_contractor_indicators

st.set_page_config(page_title="DPWH Flood-Control Screening (Projects + Contractors)", layout="wide")

# ---------- Data path (bundled; no uploads) ----------
DATA_DIR = Path(__file__).parent / "data"
BASE_CSV = DATA_DIR / "Flood_Control_Data.csv"

st.title("DPWH Flood-Control Audit")
st.caption("Rules-based screening using the DWPH and Sumbongpangulo flood control dataset; includes contractor indicators.")

with st.expander("How to read this dashboard"):
    st.markdown("""
**Purpose.** This is a *screening* tool that **prioritizes items for verification**.  
It does **not** make final findings. Always confirm with **records** (plans/estimates, POW, NTP, inspection, completion & acceptance) and **site checks**.

**Signals we compute** (tunable at left):
- **Redundant** – similar projects in the **same area & same year** (uses geo grid from Latitude/Longitude).
- **Potential “Ghost”** – status/date inconsistencies (e.g., “100% complete” but **no completion date**; **target overrun** past a grace period).
- **Never-ending** – very **long duration** (needs start+end dates), **or** recurring similar titles across **≥3 years** in the same area.
- **Costly** – outliers in **₱/km** or **₱/sq-km** (IQR method), with a clear **CostRate** and unit; falls back to IQR on **Amount** if there’s no length/area.

**Why these matter (in brief):**
- **Procurement planning & economy/efficiency** (RA 9184 IRR) discourage duplication and require needs-based projects and proper documentation.
- **Contract implementation** requires inspection, completion & acceptance files.
- **COA** (PD 1445) post-audits projects and looks for complete, reliable documentation.
See the **Legal • Data-Science • Policy** tab for links and details.
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
    help="Projects with duration ≥ this are flagged (needs start+end dates). If no end date, a stricter 'prolonged-open' rule is used."
)
cost_iqr_k = st.sidebar.slider(
    "Costly: IQR multiplier (k)", 0.5, 3.0, 1.5, step=0.1,
    help="Outlier threshold for ₱/km or ₱/sq-km via IQR. Lower k = more sensitive."
)

st.sidebar.header("Contractor Thresholds")
contr_share = st.sidebar.slider(
    "Concentration: max share in any area–year", 0.10, 0.90, 0.30, step=0.05,
    help="Flag if a contractor’s share of projects in any area–year cluster ≥ this value."
)
contr_repeat = st.sidebar.number_input(
    "Repeated issues: minimum flagged projects", min_value=1, max_value=100, value=3, step=1,
    help="Flag contractor if (ghost + never-ending + costly) flags ≥ this number."
)
contr_cost_pctl = st.sidebar.slider(
    "High mean unit cost: peer percentile", 60, 99, 90, step=1,
    help="Flag if contractor mean ₱/km (or ₱/sq-km) ≥ this percentile vs. peers."
)

# ---------- Sidebar: geographic grouping ----------
st.sidebar.header("Geographic grouping")
geo_cell_km = st.sidebar.slider(
    "Geo area cell (km)", 1, 50, 5, step=1,
    help="Projects within the same lat/lon grid cell are considered the same area (used for Redundant & Never-ending rules). Smaller cell = more precise."
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
lat_col_sel    = _sel("Latitude")
lon_col_sel    = _sel("Longitude")

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
        area=area_col_sel,
        lat=lat_col_sel,
        lon=lon_col_sel,
    ).items()
}

with st.sidebar.expander(" How these thresholds work"):
    st.markdown(f"""
- **Redundant** → Same geo cell (**{geo_cell_km} km**) & **same year** with similar titles (≥ **{redund_sim:.2f}**).  
  Use to spot probable duplicates, split-ups, or overlapping scopes that need review.
- **Potential Ghost** → High-amount (≥ **{ghost_hi_pct}**th pct) + status/date inconsistencies, incl. **target overrun** (+{grace_days}d).
- **Never-ending** → Duration ≥ **{never_days}** days (needs start+end dates), **or** recurring titles across ≥ 3 years in the same area.
- **Costly** → IQR outliers in **₱/km** or **₱/sq-km** (k = **{cost_iqr_k}**). If units missing, we temporarily use IQR on **Amount** (₱) to triage.
""")

# ---------- Cache heavy preprocessing ----------
@st.cache_data(show_spinner=False)
def _preprocess_once(df_input: pd.DataFrame, overrides: dict, geo_cell_km: float):
    return preprocess_projects(df_input, overrides=overrides, geo_cell_km=geo_cell_km)

with st.spinner("Preparing data (one-time)…"):
    prep = _preprocess_once(df, col_overrides, geo_cell_km)
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
    cols_to_show = [
        colmap.get("title") or "Title",
        "FLAG_RedundantSameAreaYear","RedundantGroupID","Reason_Redundant","RedundantPeers",
        "FLAG_PotentialGhost","Reason_PotentialGhost",
        "FLAG_NeverEnding","Reason_NeverEnding",
        "FLAG_Costly","Reason_Costly",
        "CostRate","CostRateUnit","CostRatePercentile",
        "__AreaKey","__AreaKeySource"
    ]
    for key, label in [
        ("all_flagged", "All flagged projects"),
        ("redundant", "Redundant (same area + same year + similar titles)"),
        ("ghost", "Potential ghost (status/date inconsistencies & target overrun)"),
        ("neverending", "Never-ending (long duration or recurring titles)"),
        ("costly", "Costly (₱/km or ₱/sq-km outliers; falls back to Amount if no units)"),
    ]:
        dfv = proj.get(key, pd.DataFrame())
        st.caption(f"{label} — {len(dfv):,} rows")
        show = [c for c in cols_to_show if c in dfv.columns]
        st.dataframe((dfv[show] if show else dfv).head(150), use_container_width=True)

with t2:
    st.subheader("Contractor Indicators")
    st.markdown("""
These are **screening indicators** to prioritize review. They do **not** prove wrongdoing.  
Use them to queue **document checks** (POW, plans/estimates, inspection, completion/acceptance) and **site verification**.
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
### Legal anchors (plain language, with links)
- **RA 9184 – Government Procurement Reform Act** and its **2016 Revised IRR** (as updated).  
  These set the **planning** rules (Annual Procurement Plan, needs analysis) and **contract implementation** requirements (inspection, completion, acceptance).  
  • RA 9184 (full text, GPPB)  
  • 2016 Revised IRR (GPPB; with Annexes for infra & design-build)
- **PD 1445 – Government Auditing Code**.  
  Gives **COA** authority to examine records and conduct post-audit/inspection.  
- **COA Circular 2023-004 – Updated Documentary Requirements for Common Government Transactions**.  
  Reinforces **completion/acceptance** documentation in payments.  
- **(Current landscape)** **RA 12009 – New Government Procurement Act (NGPA)** and its **IRR** were published in 2025.  
  Many principles remain (planning, competition, documentation); confirm which law/IRR applied to the **project year** in your data.
""")
    st.markdown("""
### How each project flag is computed & why it matters
- **Redundant (same area & year).**  
  We cluster by **geo grid** (Latitude/Longitude) and **calendar year**; titles must be similar (threshold at left) **and** share ≥2 uncommon tokens.  
  *Why review?* Planning & economy/efficiency principles require avoiding duplication and ensuring projects answer actual needs. Similar, same-year projects in one place can indicate possible duplication or splitter packaging that needs scrutiny.

- **Potential “Ghost”.**  
  Signals: “complete” status but **no completion date**, **very short** reported duration with **high amount**, **long-open** projects without completion, **no dates** despite **high amount**, or **target-date overrun** after a grace period.  
  *Why review?* Payments should match properly **inspected, completed, and accepted** work with documents (e.g., Inspection Report, Certificate of Completion/Acceptance). Missing or inconsistent dates for high-amount projects are red flags.

- **Never-ending.**  
  (a) **Long duration** (≥ selected days) when both **start & end** dates exist; or (b) **recurring titles** across ≥3 different years in the same area (possible repeat works without clear completion).  
  *Why review?* Extended works raise risk of variations, delays, liquidated damages, or scope creep and need closer contract-implementation checks.

- **Costly.**  
  We compute **₱/km** or **₱/sq-km** (using your length/area). We mark **IQR outliers** (k at left). If units are missing, we temporarily apply IQR to **Amount (₱)** to triage obvious anomalies.  
  *Why review?* Large over- or under-unit-costs vs. peers may indicate quantity or scope issues, mis-specification, or data errors—worthy of verification.
""")
    st.markdown("""
### Contractor indicators (screening)
- **Concentration** – A contractor’s share of projects within any geo-area & year exceeds the threshold.  
- **Repeated issues** – Count of a contractor’s projects that were flagged (ghost + never-ending + costly) ≥ threshold.  
- **High mean unit cost** – Contractor’s average **₱/km** or **₱/sq-km** sits at or above the selected **peer percentile**.

> **Reminder:** Indicators are *leads*, not findings. Always confirm with complete records and site inspection before conclusions.
""")
    st.markdown("""
### Transparency on parameters (current settings)
- Geo cell: **{geo_cell_km} km**  
- Redundant similarity: **{redund_sim:.2f}**  
- High-amount percentile: **{ghost_hi_pct}th**  
- Target overrun grace: **{grace_days} days**  
- Never-ending duration: **{never_days} days**  
- Costly IQR k: **{cost_iqr_k}**
""")
    st.markdown("""
### Quick links to the sources cited (official)
- RA 9184 (GPPB): https://www.gppb.gov.ph/wp-content/uploads/2023/06/Republic-Act-No.-9184.pdf  
- 2016 Revised IRR of RA 9184 (GPPB, updated PDF): https://www.gppb.gov.ph/wp-content/uploads/2024/07/Updated-2016-Revised-IRR-of-RA-No.-9184-as-of-19-July-2024.pdf  
- PD 1445 – Government Auditing Code (GPPB copy): https://www.gppb.gov.ph/wp-content/uploads/2023/06/Presidential-Decree-No.-1445.pdf  
- COA Circular 2023-004 (official): https://www.coa.gov.ph/wpfd_file/coa-circular-no-2023-004-june-14-2023/  
- IRR of RA 12009 (DBM/GPPB info): https://www.dbm.gov.ph/index.php/management-2/3212-irr-of-new-govt-procurement-act-now-published
""")
