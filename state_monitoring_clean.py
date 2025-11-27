import streamlit as st
import pandas as pd
from io import BytesIO

# ------------------------------------------------------------
# Helpers (no regex)
# ------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + snake_case column names."""
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
    """Try a few separators, fall back to default."""
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


def ensure_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise to columns:
    - district
    - target
    - done
    - role
    """
    df = normalize_columns(df_raw)

    district_col = pick_column(df, ["district_name", "district"])
    target_col   = pick_column(df, ["targeted_visits", "target", "visit_target", "total_target"])
    done_col     = pick_column(df, ["completed", "done", "visits_done", "achievement"])
    role_col     = pick_column(df, ["role_name", "role", "cadre", "post", "designation"])

    out = pd.DataFrame()
    out["district"] = df[district_col].astype(str).str.strip() if district_col else ""
    out["target"]   = pd.to_numeric(df[target_col], errors="coerce").fillna(0) if target_col else 0
    out["done"]     = pd.to_numeric(df[done_col], errors="coerce").fillna(0) if done_col else 0
    out["role"]     = df[role_col].astype(str).str.strip() if role_col else ""

    return out


def canon_role(s: str) -> str:
    """Canonical form for role strings."""
    s = str(s).strip().lower()
    for ch in [" ", "/", "\\", "-", ".", "(", ")", ","]:
        s = s.replace(ch, "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
    return s


# ------------------------------------------------------------
# Report builders
# ------------------------------------------------------------

def build_state_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    """District-wise overall target vs completed (all roles combined)."""
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

    g = g[["Rank", "District Name", "Total Target", "Total Completed", "% Completed"]]
    return g


def aggregate_roles(df: pd.DataFrame, role_patterns: dict) -> pd.DataFrame:
    """
    df: has columns district, target, done, role
    role_patterns: { label: [pattern1, pattern2,...] }

    Returns a df with district + for each label:
    <label> Target, <label> Completed, <label> %
    """
    df2 = df.copy()
    df2["role_key"] = df2["role"].apply(canon_role)

    base = df2[["district"]].drop_duplicates().reset_index(drop=True)

    for label, patterns in role_patterns.items():
        def match_any(x):
            for p in patterns:
                if p in x:
                    return True
            return False

        mask = df2["role_key"].apply(match_any)
        grp = df2[mask].groupby("district").agg(
            target=("target", "sum"),
            done=("done", "sum")
        ).reset_index()

        base = base.merge(grp, on="district", how="left")
        base[f"{label} Target"] = base["target"].fillna(0)
        base[f"{label} Completed"] = base["done"].fillna(0)
        base[f"{label} %"] = (
            base[f"{label} Completed"]
            / base[f"{label} Target"].replace(0, pd.NA) * 100
        ).round(2).fillna(0)

        base = base.drop(columns=["target", "done"], errors="ignore")

    return base


def build_cadre_report(df_district_raw, df_block_raw, df_cluster_raw) -> pd.DataFrame:
    """
    Cadre-wise report with SEPARATE columns for:
    - DPC
    - DIET Principal
    - DIET Academic
    - APC
    - BRCC (from BRC in block file)
    - BAC
    - CAC
    Each has Target, Completed, %.
    Also computes Total Target, Total Completed, Total % and Rank.
    """
    df_district = ensure_schema(df_district_raw)
    df_block = ensure_schema(df_block_raw)
    df_cluster = ensure_schema(df_cluster_raw)

    # --- District-level roles (based on your District.csv) ---
    # Raw role values (canon_role) are: dpc, diet_principal, apc, diet_academic
    district_roles = {
        "DPC": ["dpc"],
        "DIET Principal": ["diet_principal"],
        "DIET Academic": ["diet_academic"],
        "APC": ["apc"],
    }

    # --- Block-level roles (based on Block (2).csv) ---
    # Raw roles: BAC, BRC  -> canon: bac, brc
    block_roles = {
        "BRCC": ["brc"],  # BRC in raw treated as BRCC in report
        "BAC": ["bac"],
    }

    # --- Cluster-level roles (Cluster (3).csv) ---
    # Raw role: CAC
    cluster_roles = {
        "CAC": ["cac"],
    }

    # Aggregate each level
    dpart = aggregate_roles(df_district, district_roles)
    bpart = aggregate_roles(df_block, block_roles)
    cpart = aggregate_roles(df_cluster, cluster_roles)

    # Merge all role parts on district
    m = dpart.merge(bpart, on="district", how="outer").merge(cpart, on="district", how="outer")

    # Ensure numeric
    for c in m.columns:
        if c != "district":
            m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)

    # Overall totals across all roles
    role_labels = ["DPC", "DIET Principal", "DIET Academic", "APC", "BRCC", "BAC", "CAC"]

    total_target = 0
    total_done = 0
    for r in role_labels:
        total_target += m.get(f"{r} Target", 0)
        total_done += m.get(f"{r} Completed", 0)

    m["Total Target"] = total_target
    m["Total Completed"] = total_done
    m["Total %"] = (
        m["Total Completed"]
        / m["Total Target"].replace(0, pd.NA) * 100
    ).round(2).fillna(0)

    # Rank by Total %
    m = m.sort_values("Total %", ascending=False)
    m["Rank"] = range(1, len(m) + 1)

    m = m.rename(columns={"district": "District Name"})

    # Final column order (DIET Academic & APC separate)
    final_cols = [
        "Rank",
        "District Name",
        "DPC Target", "DPC Completed", "DPC %",
        "DIET Principal Target", "DIET Principal Completed", "DIET Principal %",
        "DIET Academic Target", "DIET Academic Completed", "DIET Academic %",
        "APC Target", "APC Completed", "APC %",
        "BRCC Target", "BRCC Completed", "BRCC %",
        "BAC Target", "BAC Completed", "BAC %",
        "CAC Target", "CAC Completed", "CAC %",
        "Total Target", "Total Completed", "Total %",
    ]
    final_cols = [c for c in final_cols if c in m.columns]
    m = m[final_cols]

    return m


# ------------------------------------------------------------
# Excel export
# ------------------------------------------------------------

def export_excel(report_df: pd.DataFrame) -> bytes:
    """
    Apply colour scale to:
    - 'Total %' (cadre-wise), OR
    - '% Completed' (state summary)
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        report_df.to_excel(writer, index=False, sheet_name="Report")

        ws = writer.sheets["Report"]
        wb = writer.book

        pct_col_name = None
        if "Total %" in report_df.columns:
            pct_col_name = "Total %"
        elif "% Completed" in report_df.columns:
            pct_col_name = "% Completed"

        if pct_col_name is not None:
            col_idx = report_df.columns.get_loc(pct_col_name)
            start = 2
            end = len(report_df) + 1
            col_letter = chr(ord("A") + col_idx)
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

        header_fmt = wb.add_format({"bold": True})
        ws.set_row(0, None, header_fmt)

    buf.seek(0)
    return buf.getvalue()


# ------------------------------------------------------------
# Streamlit UI (CSF styling)
# ------------------------------------------------------------

st.set_page_config(page_title="State Monitoring Report", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #F5F7FB;
    }
    div.block-container {
        padding-top: 3.5rem;
        padding-bottom: 1.2rem;
        max-width: 1200px;
    }
    .csf-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0054A6;
        margin-bottom: 0.4rem;
    }
    .csf-subtitle {
        font-size: 1rem;
        color: #4B4B4B;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg,#0054A6,#00A8E8);
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.45rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stButton>button:hover {
        filter: brightness(1.05);
    }
    .stSelectbox>div>div>div {
        border-radius: 999px !important;
        border-color: #0054A655 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="csf-title">State Monitoring Report</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="csf-subtitle">Upload district, block and cluster monitoring CSVs to generate a ready-to-share FLN monitoring report.</div>',
    unsafe_allow_html=True,
)

# File uploaders
c1, c2, c3 = st.columns(3)
with c1:
    f1 = st.file_uploader("District CSV (DPC / DIET / APC)", type=["csv"])
with c2:
    f2 = st.file_uploader("Block CSV (BAC / BRC)", type=["csv"])
with c3:
    f3 = st.file_uploader("Cluster CSV (CAC)", type=["csv"])

if f1 and f2 and f3:
    report_type = st.selectbox(
        "Choose report type",
        ["State Summary Report", "Cadre-wise Report"],
        index=0,
    )

    if st.button("Generate report"):
        df1_raw = read_any_csv(f1)
        df2_raw = read_any_csv(f2)
        df3_raw = read_any_csv(f3)

        # For state summary we just need standardised schema
        df1 = ensure_schema(df1_raw)
        df2 = ensure_schema(df2_raw)
        df3 = ensure_schema(df3_raw)

        if report_type == "State Summary Report":
            all_df = pd.concat([df1, df2, df3], ignore_index=True)
            report = build_state_summary(all_df)
        else:
            report = build_cadre_report(df1_raw, df2_raw, df3_raw)

        st.markdown("### Generated report")
        st.dataframe(report, use_container_width=True)

        excel_bytes = export_excel(report)
        st.download_button(
            "Download Excel report",
            excel_bytes,
            file_name="State_Monitoring_Report.xlsx",
        )

else:
    st.info("Upload all three CSVs to view and download reports.")
