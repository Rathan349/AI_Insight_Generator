import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from fpdf import FPDF
from datetime import datetime


# ── Export cleaned CSV ────────────────────────────────────────────────────────
def download_csv(df: pd.DataFrame):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Cleaned Data (CSV)",
        data=csv,
        file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── Export chart as PNG ───────────────────────────────────────────────────────
def download_chart_png(df: pd.DataFrame):
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.endswith("_normalized")]
    if not numeric_cols:
        st.warning("No numeric columns to chart.")
        return

    col = st.selectbox("Select column to export as chart", numeric_cols, key="png_col")
    chart_type = st.selectbox("Chart type", ["Histogram", "Box Plot", "Line"], key="png_type")

    fig, ax = plt.subplots(figsize=(10, 5))
    if chart_type == "Histogram":
        ax.hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Histogram: {col}")
    elif chart_type == "Box Plot":
        sns.boxplot(y=df[col].dropna(), ax=ax, color="#DD8452")
        ax.set_title(f"Box Plot: {col}")
    elif chart_type == "Line":
        ax.plot(df[col].values, color="#4C72B0", linewidth=2)
        ax.set_title(f"Line Chart: {col}")
    ax.set_xlabel(col)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()

    st.pyplot(fig)
    st.download_button(
        label="⬇️ Download Chart (PNG)",
        data=buf,
        file_name=f"chart_{col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
        use_container_width=True,
    )


# ── Copy insights to clipboard (JS trick) ────────────────────────────────────
def copy_insights_to_clipboard(insights: list[str], recs: list[str]):
    all_text = "=== AI INSIGHTS ===\n" + "\n".join(insights)
    if recs:
        all_text += "\n\n=== RECOMMENDATIONS ===\n" + "\n".join(recs)

    # Show text area so user can manually copy, plus JS auto-copy button
    st.text_area("All insights (select all & copy)", value=all_text, height=200, key="clipboard_text")
    escaped = all_text.replace("`", "'").replace("\n", "\\n")
    st.components.v1.html(f"""
        <button onclick="navigator.clipboard.writeText(`{escaped}`).then(()=>{{
            this.innerText='✅ Copied!'; setTimeout(()=>this.innerText='📋 Copy to Clipboard',2000);
        }})" style="padding:8px 18px;background:#4C72B0;color:white;border:none;
        border-radius:6px;cursor:pointer;font-size:14px;">📋 Copy to Clipboard</button>
    """, height=50)


# ── Full PDF report ───────────────────────────────────────────────────────────
def generate_pdf_report(df: pd.DataFrame, clean_report: dict, insights: list[str], recs: list[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(76, 114, 176)
    pdf.cell(0, 12, "AI Business Insight Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(6)

    # Dataset overview
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Dataset Overview", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    overview = [
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Duplicates Removed: {clean_report.get('duplicates_removed', 0)}",
        f"Missing Values Filled: {clean_report.get('missing_filled', 0)}",
        f"Cleaned Shape: {clean_report.get('cleaned_shape', df.shape)[0]} x {clean_report.get('cleaned_shape', df.shape)[1]}",
    ]
    for line in overview:
        pdf.cell(0, 6, line, ln=True)
    pdf.ln(4)

    # Summary stats
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df[[c for c in numeric_df.columns if not c.endswith("_normalized")]]
    if not numeric_df.empty:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, "Summary Statistics", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        stats = numeric_df.describe().T[["mean", "std", "min", "max"]].round(2)
        col_w = 38
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(col_w, 6, "Column", border=1)
        for h in ["Mean", "Std", "Min", "Max"]:
            pdf.cell(col_w, 6, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for col_name, row in stats.iterrows():
            pdf.cell(col_w, 6, str(col_name)[:18], border=1)
            for v in row:
                pdf.cell(col_w, 6, str(v), border=1)
            pdf.ln()
        pdf.ln(4)

    # Insights
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "AI-Generated Insights", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    for ins in insights:
        clean = ins.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, f"- {clean}")
    pdf.ln(4)

    # Recommendations
    if recs:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, "Business Recommendations", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        for rec in recs:
            clean = rec.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 6, f"- {clean}")

    return bytes(pdf.output())
