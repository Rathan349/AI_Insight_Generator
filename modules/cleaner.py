import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean dataset and return cleaned df + report."""
    report = {}
    original_shape = df.shape

    # Remove duplicates
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(dupes)

    # Handle missing values
    missing_before = df.isnull().sum().sum()
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    report["missing_filled"] = int(missing_before)

    # Normalize numeric columns (min-max)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max - col_min > 0:
            df[col + "_normalized"] = (df[col] - col_min) / (col_max - col_min)

    report["original_shape"] = original_shape
    report["cleaned_shape"] = df.shape
    report["numeric_columns"] = numeric_cols

    return df, report
