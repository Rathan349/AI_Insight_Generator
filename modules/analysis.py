import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer


def _numeric_cols(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns if not c.endswith("_normalized")]


# ── 1. Missing Value Heatmap ──────────────────────────────────────────────────
def plot_missing_heatmap(df: pd.DataFrame):
    missing = df.isnull()
    if missing.sum().sum() == 0:
        st.success("No missing values found in the dataset.")
        return
    fig, ax = plt.subplots(figsize=(12, max(4, len(df.columns) // 2)))
    sns.heatmap(missing.T, cbar=False, cmap="viridis", ax=ax, yticklabels=True)
    ax.set_title("Missing Value Heatmap (yellow = missing)", fontsize=13)
    ax.set_xlabel("Row Index")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── 2. Anomaly / Outlier Detection ───────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows as anomalies using IQR method across all numeric columns."""
    cols = _numeric_cols(df)
    if not cols:
        return df
    flags = pd.Series(False, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        flags |= (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    result = df.copy()
    result["⚠️ Anomaly"] = flags.map({True: "Yes", False: "No"})
    return result


def plot_anomalies(df: pd.DataFrame):
    cols = _numeric_cols(df)
    if not cols:
        st.warning("No numeric columns for anomaly detection.")
        return
    flagged = detect_anomalies(df)
    anomaly_count = (flagged["⚠️ Anomaly"] == "Yes").sum()
    st.metric("Anomalous Rows Detected", anomaly_count)

    col = st.selectbox("Select column to visualize anomalies", cols, key="anomaly_col")
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_outlier = (df[col] < lower) | (df[col] > upper)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df.index[~is_outlier], df[col][~is_outlier], color="#4C72B0", alpha=0.6, label="Normal", s=20)
    ax.scatter(df.index[is_outlier], df[col][is_outlier], color="#e74c3c", alpha=0.9, label="Anomaly", s=40, zorder=5)
    ax.axhline(upper, color="orange", linestyle="--", linewidth=1, label=f"Upper bound ({upper:.2f})")
    ax.axhline(lower, color="orange", linestyle="--", linewidth=1, label=f"Lower bound ({lower:.2f})")
    ax.set_title(f"Anomaly Detection: {col}")
    ax.set_xlabel("Index")
    ax.set_ylabel(col)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Flagged Records:**")
    st.dataframe(flagged[flagged["⚠️ Anomaly"] == "Yes"].head(50), use_container_width=True)


# ── 3. Time Series Detection & Trend Forecasting ─────────────────────────────
def run_time_series(df: pd.DataFrame):
    cols = _numeric_cols(df)
    if not cols:
        st.warning("No numeric columns available.")
        return

    # Try to detect datetime column
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    # Also check object cols that look like dates
    for c in df.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(df[c], infer_datetime_format=True)
            date_cols.append(c)
        except Exception:
            pass

    x_options = date_cols + ["Row Index"]
    x_col = st.selectbox("Select time/X axis", x_options, key="ts_x")
    y_col = st.selectbox("Select value column", cols, key="ts_y")
    forecast_steps = st.slider("Forecast steps ahead", 5, 50, 10, key="ts_steps")

    series = df[y_col].dropna().reset_index(drop=True)

    if x_col == "Row Index":
        x_vals = np.arange(len(series))
    else:
        try:
            x_vals = pd.to_datetime(df[x_col]).map(pd.Timestamp.toordinal).values[:len(series)]
        except Exception:
            x_vals = np.arange(len(series))

    # Linear trend fit
    coeffs = np.polyfit(x_vals, series.values, 1)
    trend_line = np.polyval(coeffs, x_vals)

    # Forecast
    future_x = np.arange(x_vals[-1] + 1, x_vals[-1] + 1 + forecast_steps)
    forecast = np.polyval(coeffs, future_x)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_vals, series.values, color="#4C72B0", label="Actual", linewidth=2)
    ax.plot(x_vals, trend_line, color="#DD8452", linestyle="--", label="Trend", linewidth=1.5)
    ax.plot(future_x, forecast, color="#55a868", linestyle="--", marker="o", markersize=4, label="Forecast")
    ax.fill_between(future_x, forecast * 0.95, forecast * 1.05, alpha=0.15, color="#55a868", label="±5% band")
    ax.set_title(f"Time Series & Forecast: {y_col}", fontsize=13)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    direction = "upward 📈" if coeffs[0] > 0 else "downward 📉"
    st.info(f"Trend direction: {direction} | Slope: {coeffs[0]:.4f} per step")


# ── 4. K-Means Clustering ─────────────────────────────────────────────────────
def run_clustering(df: pd.DataFrame):
    cols = _numeric_cols(df)
    if len(cols) < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
        return

    n_clusters = st.slider("Number of clusters (K)", 2, 10, 3, key="kmeans_k")
    x_col = st.selectbox("X axis", cols, key="km_x")
    y_col = st.selectbox("Y axis", cols, index=min(1, len(cols) - 1), key="km_y")

    X = df[cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    result = df.copy()
    result["Cluster"] = labels.astype(str)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("tab10", n_clusters)
    for i in range(n_clusters):
        mask = result["Cluster"] == str(i)
        ax.scatter(result.loc[mask, x_col], result.loc[mask, y_col],
                   label=f"Cluster {i}", color=palette[i], alpha=0.7, s=40)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"K-Means Clustering (K={n_clusters})", fontsize=13)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Cluster Summary:**")
    summary = result.groupby("Cluster")[cols].mean().round(2)
    st.dataframe(summary, use_container_width=True)


# ── 5. Feature Importance ─────────────────────────────────────────────────────
def run_feature_importance(df: pd.DataFrame):
    cols = _numeric_cols(df)
    if len(cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        return

    target = st.selectbox("Select target column", cols, key="fi_target")
    features = [c for c in cols if c != target]

    X = df[features].copy()
    y = df[target].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    y_imp = SimpleImputer(strategy="median").fit_transform(y.values.reshape(-1, 1)).ravel()

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_imp, y_imp)
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.4)))
    colors = sns.color_palette("Blues_r", len(importances))
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Feature Importance → predicting '{target}'", fontsize=13)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    top = importances.iloc[-1]
    st.info(f"Most important feature: **{importances.index[-1]}** (score: {top:.4f})")
