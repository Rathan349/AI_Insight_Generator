import pandas as pd
import numpy as np


def generate_insights(df: pd.DataFrame, clean_report: dict) -> list[str]:
    """Generate plain-English business insights from the dataset."""
    insights = []
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.endswith("_normalized")]

    if not numeric_cols:
        return ["No numeric columns found for insight generation."]

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        mean_val = series.mean()
        max_val = series.max()
        min_val = series.min()
        std_val = series.std()
        median_val = series.median()

        insights.append(f"📌 [{col}] Average value is {mean_val:.2f}, ranging from {min_val:.2f} to {max_val:.2f}.")

        # Skewness insight
        skew = series.skew()
        if abs(skew) > 1:
            direction = "right (positively)" if skew > 0 else "left (negatively)"
            insights.append(f"📐 [{col}] Distribution is skewed {direction} — outliers may be present.")

        # Variability insight
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0
        if cv > 50:
            insights.append(f"⚠️ [{col}] High variability detected (CV={cv:.1f}%) — data is widely spread.")
        elif cv < 10:
            insights.append(f"✅ [{col}] Low variability (CV={cv:.1f}%) — values are consistent.")

        # Top value insight
        if df[col].dtype in [np.int64, np.float64]:
            top_idx = df[col].idxmax()
            insights.append(f"🏆 [{col}] Peak value of {max_val:.2f} found at record index {top_idx}.")

    # Trend detection for time-like or sequential data
    for col in numeric_cols:
        series = df[col].dropna().reset_index(drop=True)
        if len(series) > 10:
            first_half = series[: len(series) // 2].mean()
            second_half = series[len(series) // 2 :].mean()
            if first_half > 0:
                change_pct = ((second_half - first_half) / first_half) * 100
                if abs(change_pct) > 10:
                    direction = "increased" if change_pct > 0 else "decreased"
                    insights.append(
                        f"📈 [{col}] Values {direction} by {abs(change_pct):.1f}% in the second half of the dataset."
                    )

    # Categorical insights
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols[:3]:  # limit to first 3 categorical cols
        top_val = df[col].value_counts().idxmax()
        top_count = df[col].value_counts().max()
        pct = (top_count / len(df)) * 100
        insights.append(f"🔖 [{col}] Most frequent value is '{top_val}' ({pct:.1f}% of records).")

    # Cleaning summary
    if clean_report.get("duplicates_removed", 0) > 0:
        insights.append(f"🧹 {clean_report['duplicates_removed']} duplicate records were removed during cleaning.")
    if clean_report.get("missing_filled", 0) > 0:
        insights.append(f"🔧 {clean_report['missing_filled']} missing values were auto-filled using median/mode imputation.")

    return insights


def generate_recommendations(df: pd.DataFrame) -> list[str]:
    """Generate business recommendations based on data patterns."""
    recs = []
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.endswith("_normalized")]

    if not numeric_cols:
        return recs

    # High-value column recommendation
    means = {col: df[col].mean() for col in numeric_cols}
    top_col = max(means, key=means.get)
    recs.append(f"💡 Focus attention on '{top_col}' — it has the highest average value and may drive key metrics.")

    # Outlier recommendation
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        if len(outliers) > 0:
            pct = (len(outliers) / len(df)) * 100
            recs.append(f"🔍 '{col}' has {len(outliers)} outliers ({pct:.1f}%) — investigate these records for anomalies.")

    # Low variance recommendation
    for col in numeric_cols:
        if df[col].std() == 0:
            recs.append(f"⚠️ '{col}' has zero variance — consider removing this column as it adds no analytical value.")

    return recs
