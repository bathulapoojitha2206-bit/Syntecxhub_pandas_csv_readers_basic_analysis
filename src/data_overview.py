"""
Dataset Overview and Statistical Profiling Module.
Provides functions for dataset metrics, column profiling, head/tail/sample previews,
memory consumption analysis, and comprehensive numeric and categorical statistical summaries.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def format_bytes(size_bytes: int) -> str:
    """Formats raw byte counts into human-readable strings (B, KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_dataset_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level key performance indicators (KPIs) for the dataset.
    """
    if df is None or df.empty:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "numeric_columns_count": 0,
            "categorical_columns_count": 0,
            "datetime_columns_count": 0,
            "total_missing_values": 0,
            "missing_cell_percentage": 0.0,
            "duplicate_rows_count": 0,
            "duplicate_percentage": 0.0,
            "memory_usage_bytes": 0,
            "memory_usage_formatted": "0 B",
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    total_missing = int(df.isna().sum().sum())
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())
    dup_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0

    mem_bytes = int(df.memory_usage(deep=True).sum())
    mem_formatted = format_bytes(mem_bytes)

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "numeric_columns_count": len(numeric_cols),
        "categorical_columns_count": len(cat_cols),
        "datetime_columns_count": len(datetime_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "datetime_columns": datetime_cols,
        "total_missing_values": total_missing,
        "missing_cell_percentage": round(missing_pct, 2),
        "duplicate_rows_count": duplicate_rows,
        "duplicate_percentage": round(dup_pct, 2),
        "memory_usage_bytes": mem_bytes,
        "memory_usage_formatted": mem_formatted,
    }


def get_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a comprehensive column profile table with data types, nulls,
    uniqueness, sample values, and memory usage.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    total_rows = len(df)
    profiles = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))

        # Sample unique values (first 3-4 distinct values)
        unique_vals = series.dropna().unique()
        if len(unique_vals) > 0:
            sample_str = ", ".join(map(str, unique_vals[:3]))
            if len(unique_vals) > 3:
                sample_str += "..."
        else:
            sample_str = "None"

        try:
            mem_col = int(series.memory_usage(deep=True))
            mem_str = format_bytes(mem_col)
        except Exception:
            mem_str = "N/A"

        profiles.append({
            "Column Name": col,
            "Data Type": dtype,
            "Non-Null Count": non_null_count,
            "Missing Count": missing_count,
            "Missing %": missing_pct,
            "Unique Count": unique_count,
            "Sample Values": sample_str,
            "Memory Usage": mem_str,
        })

    return pd.DataFrame(profiles)


def get_head(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Returns the first n rows using head()."""
    return df.head(n)


def get_tail(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Returns the last n rows using tail()."""
    return df.tail(n)


def get_sample(df: pd.DataFrame, n: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Returns a random sample of n rows using sample()."""
    if df is None or df.empty:
        return pd.DataFrame()
    sample_size = min(n, len(df))
    return df.sample(n=sample_size, random_state=random_state)


def get_numerical_summary(
    df: pd.DataFrame, selected_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculates detailed summary statistics for numeric columns:
    Count, Mean, Median, Mode, Std, Variance, Min, Max, Range,
    Quartiles (25%, 50%, 75%), Percentiles (5%, 95%), Skewness, Kurtosis, and Sum.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    num_df = df.select_dtypes(include=[np.number])
    if selected_columns:
        num_cols = [c for c in selected_columns if c in num_df.columns]
        num_df = num_df[num_cols]

    if num_df.empty:
        return pd.DataFrame()

    stats_list = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0:
            continue

        c_mean = float(s.mean())
        c_median = float(s.median())
        modes = s.mode()
        c_mode = float(modes.iloc[0]) if not modes.empty else np.nan
        c_min = float(s.min())
        c_max = float(s.max())
        c_std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        c_var = float(s.var(ddof=1)) if len(s) > 1 else 0.0
        c_sum = float(s.sum())
        c_range = c_max - c_min
        c_q25 = float(s.quantile(0.25))
        c_q50 = float(s.quantile(0.50))
        c_q75 = float(s.quantile(0.75))
        c_iqr = c_q75 - c_q25
        c_p05 = float(s.quantile(0.05))
        c_p95 = float(s.quantile(0.95))
        c_skew = float(s.skew()) if len(s) > 2 else np.nan
        c_kurt = float(s.kurt()) if len(s) > 3 else np.nan

        stats_list.append({
            "Metric": col,
            "Count": int(len(s)),
            "Mean": round(c_mean, 2),
            "Median": round(c_median, 2),
            "Mode": round(c_mode, 2) if not np.isnan(c_mode) else "N/A",
            "Std Dev": round(c_std, 2),
            "Variance": round(c_var, 2),
            "Min": round(c_min, 2),
            "Max": round(c_max, 2),
            "Range": round(c_range, 2),
            "Q25 (25%)": round(c_q25, 2),
            "Q50 (Median)": round(c_q50, 2),
            "Q75 (75%)": round(c_q75, 2),
            "IQR": round(c_iqr, 2),
            "5th Pct": round(c_p05, 2),
            "95th Pct": round(c_p95, 2),
            "Skewness": round(c_skew, 2) if not np.isnan(c_skew) else "N/A",
            "Kurtosis": round(c_kurt, 2) if not np.isnan(c_kurt) else "N/A",
            "Sum": round(c_sum, 2),
        })

    if not stats_list:
        return pd.DataFrame()

    res_df = pd.DataFrame(stats_list).set_index("Metric").T
    return res_df


def get_categorical_summary(
    df: pd.DataFrame, selected_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculates summary statistics for categorical columns:
    Count, Non-Null Count, Missing Count, Unique Values, Top (Most Frequent),
    Top Frequency, Top %, Least Frequent.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cat_df = df.select_dtypes(exclude=[np.number, "datetime", "datetimetz"])
    if selected_columns:
        cat_cols = [c for c in selected_columns if c in cat_df.columns]
        cat_df = cat_df[cat_cols]

    if cat_df.empty:
        return pd.DataFrame()

    summary_list = []
    total_rows = len(df)

    for col in cat_df.columns:
        s = cat_df[col].astype(str)
        non_null_count = int(df[col].notna().sum())
        missing_count = int(df[col].isna().sum())
        val_counts = df[col].dropna().value_counts()
        unique_count = len(val_counts)

        if unique_count > 0:
            top_val = str(val_counts.index[0])
            top_freq = int(val_counts.iloc[0])
            top_pct = round((top_freq / non_null_count * 100), 2) if non_null_count > 0 else 0.0
            least_val = str(val_counts.index[-1])
            least_freq = int(val_counts.iloc[-1])
        else:
            top_val, top_freq, top_pct = "N/A", 0, 0.0
            least_val, least_freq = "N/A", 0

        summary_list.append({
            "Column Name": col,
            "Total Rows": total_rows,
            "Non-Null Count": non_null_count,
            "Missing Count": missing_count,
            "Unique Count": unique_count,
            "Most Frequent": top_val,
            "Most Freq Count": top_freq,
            "Most Freq %": top_pct,
            "Least Frequent": least_val,
            "Least Freq Count": least_freq,
        })

    return pd.DataFrame(summary_list)


def get_column_value_counts(
    df: pd.DataFrame, column: str, top_n: int = 15
) -> pd.DataFrame:
    """
    Returns value counts and frequency percentages for a single column.
    """
    if df is None or column not in df.columns:
        return pd.DataFrame()

    series = df[column].dropna()
    total = len(series)
    if total == 0:
        return pd.DataFrame(columns=[column, "Count", "Percentage"])

    vc = series.value_counts().head(top_n).reset_index()
    vc.columns = [column, "Count"]
    vc["Percentage"] = (vc["Count"] / total * 100).round(2)
    return vc
