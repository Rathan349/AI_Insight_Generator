import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def plot_correlation_matrix(df: pd.DataFrame):
    """Heatmap of correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df[[c for c in numeric_df.columns if not c.endswith("_normalized")]]

    if numeric_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr.columns)), max(6, len(corr.columns) - 1)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, linewidths=0.5, square=True
    )
    ax.set_title("Correlation Matrix", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def get_strong_correlations(df: pd.DataFrame, threshold: float = 0.6) -> list[dict]:
    """Return pairs of strongly correlated features."""
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df[[c for c in numeric_df.columns if not c.endswith("_normalized")]]
    corr = numeric_df.corr()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({"col_a": cols[i], "col_b": cols[j], "correlation": round(val, 3)})
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)


def plot_top_feature_relationships(df: pd.DataFrame, top_n: int = 3):
    """Scatter plots for top correlated pairs."""
    pairs = get_strong_correlations(df, threshold=0.4)[:top_n]
    if not pairs:
        st.info("No strong feature relationships found (threshold: 0.4).")
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        ax.scatter(df[pair["col_a"]], df[pair["col_b"]], alpha=0.5, color="#DD8452")
        ax.set_xlabel(pair["col_a"])
        ax.set_ylabel(pair["col_b"])
        ax.set_title(f'r = {pair["correlation"]}', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
