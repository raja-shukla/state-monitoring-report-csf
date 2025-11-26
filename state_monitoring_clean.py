import streamlit as st
import pandas as pd
from io import BytesIO

# ------------------------------------------------------------
# Helpers (no regex)
# ------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    new_cols = []
    for c in df.columns:
        s = str(c).strip().lower()
        for ch in [" ", "/", "\\", "-", ".", "(", ")", ":", ","]:
            s = s.replace(ch, "_")
        s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
        new_cols.append(s)
    df2.columns = new_cols
    return df2


def read_any_csv(upload) -> pd.DataFrame:
    raw = upload.read()
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(BytesIO(raw), sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(BytesIO(raw))


def pick_column(df: pd.DataFrame, candidates):
    cols = df.columns
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        for col in cols:
            if c in col:
                return col
    return None


def ensure_schema(df_raw):
    df = normalize_columns(df_raw)

    district_col = pick_column(df, ["district", "jila", "zilla"])
    target_col   = pick_column(df, ["target", "visit_target", "total_target"])
    done_col     = pick_column(df, ["done", "completed", "visits_done", "achievement"])

    out = pd.DataFrame()
    out["district"] = df[district_col].astype(str).str.strip() if district_col else ""
    out["target"]   = pd.to_numeric(df[target_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0) if target_col else 0
    out["done"]     = pd.to_numeric(df[done_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0) if done_col else 0

    return out


# ------------------------------------------------------------
# Report builders
# ------------------------------------------------------------

def build_state_summary(all_df):
    g = all_df.groupby("district").agg(
        total_visit_target=("target", "sum"),
        completed_visit=("done", "sum")
    ).reset_index()

    g["% Completed"] = (
        g["completed_visit"] / g["total_visit_target"].replace(0, pd.NA) * 100
    ).round(2).fillna(0)

    g = g.sort_values("% Completed", ascending=False)
    g["Rank"] = range(1, len(g) + 1)

    g = g.rename(columns={
        "district": "District Name",
        "total_visit_target": "Total Target",
        "completed_visit": "Total Completed"
    })

    return g


def _agg(df, label):
    g = df.groupby("district").agg(
        target=("target", "sum"),
        done=("done", "sum"),
    ).reset_index()

    g[f"{label} Target"] = g["target"]
    g[f"{label} Completed"] = g["done"]
    g[f"{label} %"] = (g["done"] / g["target"].replace(0, pd.NA) * 100).round(2).fillna(0)

    return g[["district", f"{label} Target", f"{label} Completed", f"{label} %"]]


def build_cadre_report(df_district, df_block, df_cluster):
    dpc = _agg(df_district, "DPC+APC+DIET")
    brc = _agg(df_block,    "BAC+BRC")
    cac = _agg(df_cluster,  "CAC")

    m = dpc.merge(brc, on="district", how="outer").merge(cac, on="district", how="outer")

    for c in m.columns:
        if c != "district":
            m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)

    m["Total Target"] = m["DPC+APC+DIET Target"] + m["BAC+BRC Target"] + m["CAC Target"]
    m["Total Completed"] = m["DPC+APC+DIET Completed"] + m["BAC+BRC Completed"] + m["CAC Completed"]
    m["Total %"] = (m["Total Completed"] / m["Total Target"].replace(0, pd.NA) * 100).round(2).fillna(0)

    m = m.sort_values("Total %", ascending=False)
    m["Rank"] = range(1, len(m) + 1)

    m = m.rename(columns={"district": "District Name"})

    final_cols = [
        "Rank",
        "District Name",
        "DPC+APC+DIET %",
        "BAC+BRC Target", "BAC+BRC Completed", "BAC+BRC %",
        "CAC %",
        "Total Target", "Total Completed", "Total %",
    ]

    m = m[final_cols]

    return m


# ------------------------------------------------------------
# Excel export
# ------------------------------------------------------------

def export_excel(report_df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        report_df.to_excel(writer, index=False, sheet_name="Report")

        ws = writer.sheets["Report"]
        wb = writer.book

        # 3-colour scale on Total %
        if "Total %" in report_df.columns:
            col = report_df.columns.get_loc("Total %")
            start = 2
            end = len(report_df) + 1
            col_letter = chr(ord("A") + col)
            ws.conditional_format(
                f"{col_letter}{start}:{col_letter}{end}",
                {
                    "type": "3_color_scale",
                    "min_type": "min",
                    "mid_type": "percentile",
                    "mid_value": 50,
                    "max_type": "max",
                },
            )

        # Bold header
        header_fmt = wb.add_format({"bold": True})
        ws.set_row(0, None, header_fmt)

    buf.seek(0)
    return buf.getvalue()


# ------------------------------------------------------------
# Streamlit UI (CSF-ish styling)
# ------------------------------------------------------------

st.set_page_config(
    page_title="State Monitoring Report",
    page_icon=None,
    layout="wide"
)

# Custom CSS – CSF-style colours (deep blue + teal + clean background)
st.markdown(
    """
    <style>
    .main {
        background-color: #F5F7FB;
    }
    div.block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 1200px;
    }
    .csf-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0054A6; /* deep blue */
        margin-bottom: 0.15rem;
    }
    .csf-subtitle {
        font-size: 0.95rem;
        color: #4B4B4B;
        margin-bottom: 1.1rem;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#0054A6,#00A8E8);
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.4rem 1.4rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .stButton>button:hover {
        filter: brightness(1.05);
    }
    /* Selectbox border rounding */
    .stSelectbox>div>div>div {
        border-radius: 999px !important;
        border-color: #0054A655 !important;
    }
    /* File uploader card feel */
    .uploadedFile, .stFileUploader>div>div {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="csf-title">State Monitoring Report</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="csf-subtitle">'
    'Upload the latest district, block and cluster monitoring CSVs and generate a clean, ready-to-share report.'
    '</div>',
    unsafe_allow_html=True,
)

# Uploaders
c1, c2, c3 = st.columns(3)
with c1:
    f1 = st.file_uploader("District CSV (DPC/APC/DIET)", type=["csv"])
with c2:
    f2 = st.file_uploader("Block CSV (BAC/BRC)", type=["csv"])
with c3:
    f3 = st.file_uploader("Cluster CSV (CAC)", type=["csv"])

if f1 and f2 and f3:
    st.markdown("### Report configuration")
    report_type = st.selectbox(
        "Choose report type",
        ["State Summary Report", "Cadre-wise Report"],
        index=0,
    )

    if st.button("Generate report"):
        df1 = ensure_schema(read_any_csv(f1))
        df2 = ensure_schema(read_any_csv(f2))
        df3 = ensure_schema(read_any_csv(f3))

        all_df = pd.concat([df1, df2, df3], ignore_index=True)

        if report_type == "State Summary Report":
            report = build_state_summary(all_df)
        else:
            report = build_cadre_report(df1, df2, df3)

        st.markdown("### Generated report")
        st.dataframe(report, use_container_width=True)

        excel_bytes = export_excel(report)
        st.download_button(
            "Download Excel report",
            excel_bytes,
            file_name="State_Monitoring_Report.xlsx",
            type="primary",
        )
else:
    st.info("Upload all three CSVs to view and download reports.")
