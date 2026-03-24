import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules.cleaner import clean_data
from modules.eda import get_summary_stats, plot_distributions, plot_data_types, plot_pie_charts, plot_pairplot, plot_custom_builder
from modules.patterns import plot_correlation_matrix, get_strong_correlations, plot_top_feature_relationships
from modules.insights import generate_insights, generate_recommendations
from modules.analysis import (
    plot_missing_heatmap, detect_anomalies, plot_anomalies,
    run_time_series, run_clustering, run_feature_importance
)
from modules.exporter import download_csv, download_chart_png, copy_insights_to_clipboard, generate_pdf_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Business Insight Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #4C72B0, #DD8452);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #888; font-size: 1rem; margin-bottom: 1.5rem; }
    .insight-card {
        background: #1e1e2e; border-left: 4px solid #4C72B0;
        padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .rec-card {
        background: #1e2e1e; border-left: 4px solid #55a868;
        padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .metric-box {
        background: #1e1e2e; border-radius: 8px;
        padding: 1rem; text-align: center;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.markdown("## 📊 AI Insight Generator")
    st.markdown("---")

    st.markdown("### 📥 Data Input")
    input_method = st.radio(
        "Choose input method",
        ["📂 Upload File", "📋 Paste from Clipboard", "🔗 Google Sheets URL"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    clipboard_data = None
    sheets_url = None

    if input_method == "📂 Upload File":
        uploaded_file = st.file_uploader("Upload CSV, Excel or JSON", type=["csv", "xlsx", "xls", "json"])
    elif input_method == "📋 Paste from Clipboard":
        clipboard_data = st.text_area("Paste CSV data here", height=150, placeholder="col1,col2,col3\n1,2,3\n4,5,6")
    elif input_method == "🔗 Google Sheets URL":
        sheets_url = st.text_input("Google Sheets URL", placeholder="https://docs.google.com/spreadsheets/d/...")
        st.caption("Sheet must be publicly shared (Anyone with link can view)")

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")
    show_raw = st.checkbox("Show Raw Data Preview", value=True)
    show_cleaned = st.checkbox("Show Cleaned Data Info", value=True)
    show_eda = st.checkbox("Show EDA & Distributions", value=True)
    show_patterns = st.checkbox("Show Pattern & Correlation Analysis", value=True)
    show_insights = st.checkbox("Show AI Insights", value=True)
    show_recs = st.checkbox("Show Business Recommendations", value=True)

    corr_threshold = st.slider("Correlation Threshold", 0.1, 1.0, 0.6, 0.05)

    st.markdown("---")
    st.markdown("**Supported formats:** CSV, Excel, JSON, Google Sheets")
    st.markdown("**Version:** 2.0.0")

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">AI-Powered Business Insight Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your dataset and get instant AI-driven analysis, patterns, and insights.</div>', unsafe_allow_html=True)

if uploaded_file is None and not clipboard_data and not sheets_url:
    st.info("👈 Choose a data input method from the sidebar to get started.")
    st.markdown("### What this tool does:")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**🧹 Auto Data Cleaning**\n\nRemoves duplicates, fills missing values, normalizes numeric data.")
    with cols[1]:
        st.markdown("**📊 Exploratory Analysis**\n\nDistribution plots, summary stats, and data type breakdown.")
    with cols[2]:
        st.markdown("**🤖 AI Insights**\n\nPlain-English summaries, trend detection, and business recommendations.")
    st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_file(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif name.endswith(".json"):
        return pd.read_json(file)
    raise ValueError(f"Unsupported file type: {file.name}")

@st.cache_data
def load_clipboard(text: str) -> pd.DataFrame:
    import io
    return pd.read_csv(io.StringIO(text))

@st.cache_data
def load_google_sheet(url: str) -> pd.DataFrame:
    # Convert share URL to CSV export URL
    if "/edit" in url:
        url = url.split("/edit")[0]
    elif url.endswith("/"):
        url = url.rstrip("/")
    # Extract sheet ID and build export link
    import re
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise ValueError("Could not extract Sheet ID from URL. Make sure it's a valid Google Sheets link.")
    sheet_id = match.group(1)
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    return pd.read_csv(export_url)

try:
    if uploaded_file:
        raw_df = load_file(uploaded_file)
    elif clipboard_data:
        raw_df = load_clipboard(clipboard_data)
    elif sheets_url:
        with st.spinner("Fetching Google Sheet..."):
            raw_df = load_google_sheet(sheets_url)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ── Raw Preview ───────────────────────────────────────────────────────────────
if show_raw:
    st.markdown("## 📂 Dataset Preview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", raw_df.shape[0])
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Missing Values", int(raw_df.isnull().sum().sum()))
    c4.metric("Duplicates", int(raw_df.duplicated().sum()))
    st.dataframe(raw_df.head(10), use_container_width=True)

# ── Clean Data ────────────────────────────────────────────────────────────────
with st.spinner("🧹 Cleaning data..."):
    df, clean_report = clean_data(raw_df.copy())

if show_cleaned:
    st.markdown("## 🧹 Data Cleaning Report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicates Removed", clean_report["duplicates_removed"])
    c2.metric("Missing Values Filled", clean_report["missing_filled"])
    c3.metric("Cleaned Shape", f"{clean_report['cleaned_shape'][0]} × {clean_report['cleaned_shape'][1]}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 EDA", "🔍 Patterns", "🔬 Analysis", "🎨 Visualizations", "🤖 AI Insights", "📈 Recommendations", "📤 Export"])

# Tab 1: EDA
with tabs[0]:
    if show_eda:
        st.markdown("### Summary Statistics")
        stats = get_summary_stats(df)
        if not stats.empty:
            st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True)
        else:
            st.warning("No numeric columns for statistics.")

        st.markdown("### Column Data Types")
        plot_data_types(df)

        st.markdown("### Distributions")
        plot_distributions(df)
    else:
        st.info("EDA is disabled. Enable it from the sidebar.")

# Tab 2: Patterns
with tabs[1]:
    if show_patterns:
        st.markdown("### Correlation Matrix")
        plot_correlation_matrix(df)

        st.markdown("### Top Feature Relationships")
        plot_top_feature_relationships(df, top_n=3)

        st.markdown("### Strong Correlations")
        pairs = get_strong_correlations(df, threshold=corr_threshold)
        if pairs:
            pairs_df = pd.DataFrame(pairs)
            st.dataframe(pairs_df, use_container_width=True)
        else:
            st.info(f"No correlations found above threshold {corr_threshold}.")
    else:
        st.info("Pattern analysis is disabled. Enable it from the sidebar.")

# Tab 3: Analysis
with tabs[2]:
    analysis_section = st.selectbox(
        "Choose analysis type",
        ["🗺️ Missing Value Heatmap", "⚠️ Anomaly Detection", "📈 Time Series & Forecast", "🔵 K-Means Clustering", "🏆 Feature Importance"],
        key="analysis_type"
    )
    if analysis_section == "🗺️ Missing Value Heatmap":
        st.markdown("### Missing Value Heatmap")
        plot_missing_heatmap(raw_df)
    elif analysis_section == "⚠️ Anomaly Detection":
        st.markdown("### Anomaly / Outlier Detection")
        plot_anomalies(df)
    elif analysis_section == "📈 Time Series & Forecast":
        st.markdown("### Time Series Detection & Trend Forecast")
        run_time_series(df)
    elif analysis_section == "🔵 K-Means Clustering":
        st.markdown("### K-Means Clustering")
        run_clustering(df)
    elif analysis_section == "🏆 Feature Importance":
        st.markdown("### Feature Importance Ranking")
        run_feature_importance(df)

# Tab 4: Visualizations
with tabs[3]:
    viz_section = st.selectbox(
        "Choose visualization",
        ["🥧 Pie / Donut Charts", "🔲 Scatter Plot Matrix (Pairplot)", "🛠️ Custom Chart Builder", "🗺️ Missing Value Heatmap"],
        key="viz_type"
    )
    if viz_section == "🥧 Pie / Donut Charts":
        st.markdown("### Pie / Donut Charts")
        plot_pie_charts(df)
    elif viz_section == "🔲 Scatter Plot Matrix (Pairplot)":
        st.markdown("### Scatter Plot Matrix")
        plot_pairplot(df)
    elif viz_section == "🛠️ Custom Chart Builder":
        st.markdown("### Custom Chart Builder")
        plot_custom_builder(df)
    elif viz_section == "🗺️ Missing Value Heatmap":
        st.markdown("### Missing Value Heatmap")
        plot_missing_heatmap(raw_df)

# Tab 5: AI Insights
with tabs[4]:
    if show_insights:
        st.markdown("### 🤖 Auto-Generated Insights")
        with st.spinner("Generating insights..."):
            insights = generate_insights(df, clean_report)
        for insight in insights:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
    else:
        st.info("Insights are disabled. Enable them from the sidebar.")

# Tab 6: Recommendations
with tabs[5]:
    if show_recs:
        st.markdown("### 💡 Business Recommendations")
        with st.spinner("Generating recommendations..."):
            recs = generate_recommendations(df)
        if recs:
            for rec in recs:
                st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)
        else:
            st.info("No recommendations generated for this dataset.")
    else:
        st.info("Recommendations are disabled. Enable them from the sidebar.")

# Tab 7: Export
with tabs[6]:
    st.markdown("### 📤 Export & Sharing")

    st.markdown("#### ⬇️ Export Cleaned Data")
    download_csv(df)

    st.markdown("---")
    st.markdown("#### 🖼️ Export Chart as PNG")
    download_chart_png(df)

    st.markdown("---")
    st.markdown("#### 📋 Copy Insights to Clipboard")
    with st.spinner("Preparing insights..."):
        _insights = generate_insights(df, clean_report)
        _recs = generate_recommendations(df)
    copy_insights_to_clipboard(_insights, _recs)

    st.markdown("---")
    st.markdown("#### 📄 Download Full PDF Report")
    if st.button("Generate PDF Report", use_container_width=True):
        with st.spinner("Building PDF..."):
            _insights = generate_insights(df, clean_report)
            _recs = generate_recommendations(df)
            pdf_bytes = generate_pdf_report(df, clean_report, _insights, _recs)
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"insight_report_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#555;font-size:0.85rem;'>AI Business Insight Generator • Built with Streamlit & Python</center>",
    unsafe_allow_html=True,
)
