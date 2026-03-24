import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df.select_dtypes(include=[np.number]).describe().T


def plot_distributions(df: pd.DataFrame):
    """Plot histograms for all numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude normalized columns for cleaner display
    numeric_cols = [c for c in numeric_cols if not c.endswith("_normalized")]

    if not numeric_cols:
        st.warning("No numeric columns found for distribution plots.")
        return

    cols_per_row = 3
    rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#4C72B0")
        axes[i].set_title(f"Distribution: {col}", fontsize=11)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_data_types(df: pd.DataFrame):
    """Bar chart of column data types."""
    type_counts = df.dtypes.astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    type_counts.plot(kind="bar", ax=ax, color=sns.color_palette("pastel"))
    ax.set_title("Column Data Types")
    ax.set_xlabel("Data Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_boxplots(df: pd.DataFrame):
    """Box plots to visualize outliers for numeric columns."""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.endswith("_normalized")]
    if not numeric_cols:
        st.warning("No numeric columns for box plots.")
        return

    cols_per_row = 3
    rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color="#DD8452")
        axes[i].set_title(f"Outliers: {col}", fontsize=11)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_custom_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str):
    """Render a user-defined line or bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if chart_type == "Line Chart":
        ax.plot(df[x_col], df[y_col], marker="o", color="#4C72B0", linewidth=2)
    elif chart_type == "Bar Chart":
        ax.bar(df[x_col].astype(str), df[y_col], color="#4C72B0")
        plt.xticks(rotation=45, ha="right")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{chart_type}: {y_col} vs {x_col}", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_pie_charts(df: pd.DataFrame):
    """Pie/donut charts for categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.warning("No categorical columns found.")
        return
    col_select = st.multiselect("Select categorical columns", cat_cols, default=cat_cols[:min(3, len(cat_cols))], key="pie_cols")
    if not col_select:
        return
    cols_per_row = 3
    rows = (len(col_select) + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(6 * cols_per_row, 5 * rows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(col_select):
        counts = df[col].value_counts().head(10)
        wedges, texts, autotexts = axes[i].pie(
            counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.6), colors=sns.color_palette("tab10", len(counts))
        )
        for at in autotexts:
            at.set_fontsize(8)
        axes[i].set_title(f"{col} (top {len(counts)})", fontsize=11)
    for j in range(len(col_select), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_pairplot(df: pd.DataFrame):
    """Scatter plot matrix for numeric columns."""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.endswith("_normalized")]
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for pairplot.")
        return
    selected = st.multiselect("Select columns for pairplot", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))], key="pair_cols")
    if len(selected) < 2:
        st.info("Select at least 2 columns.")
        return
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    hue_col = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="pair_hue")
    subset = df[selected].copy()
    if hue_col != "None":
        subset[hue_col] = df[hue_col].astype(str)
    with st.spinner("Generating pairplot..."):
        fig = sns.pairplot(subset, hue=hue_col if hue_col != "None" else None,
                           diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20}, palette="tab10")
        fig.fig.suptitle("Scatter Plot Matrix", y=1.02, fontsize=13)
        st.pyplot(fig.fig)
        plt.close()


def plot_custom_builder(df: pd.DataFrame):
    """Interactive custom chart builder."""
    all_cols = df.columns.tolist()
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.endswith("_normalized")]
    c1, c2, c3 = st.columns(3)
    with c1:
        chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Area Chart", "Histogram"], key="cb_type")
    with c2:
        x_col = st.selectbox("X Axis", all_cols, key="cb_x")
    with c3:
        y_col = st.selectbox("Y Axis / Column", numeric_cols, key="cb_y")
    color = st.color_picker("Chart color", "#4C72B0", key="cb_color")
    fig, ax = plt.subplots(figsize=(11, 5))
    try:
        if chart_type == "Bar Chart":
            ax.bar(df[x_col].astype(str), df[y_col], color=color)
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "Line Chart":
            ax.plot(df[x_col], df[y_col], color=color, linewidth=2, marker="o", markersize=3)
        elif chart_type == "Scatter Plot":
            ax.scatter(df[x_col], df[y_col], color=color, alpha=0.6, s=30)
        elif chart_type == "Area Chart":
            ax.fill_between(range(len(df)), df[y_col], color=color, alpha=0.5)
            ax.plot(range(len(df)), df[y_col], color=color, linewidth=1.5)
        elif chart_type == "Histogram":
            ax.hist(df[y_col].dropna(), bins=30, color=color, edgecolor="white")
            ax.set_ylabel("Frequency")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{chart_type}: {y_col} vs {x_col}", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not render chart: {e}")
    finally:
        plt.close()
