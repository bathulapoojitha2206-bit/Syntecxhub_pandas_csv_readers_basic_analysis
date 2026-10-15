"""
Data Quality and Cleaning Module.
Provides interactive cleaning operations including missing value detection and imputation,
duplicate detection and deduplication, safe type casting, negative value validation,
and IQR-based outlier detection with user-guided handling.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies missing values across all columns.
    Returns a DataFrame with Column Name, Missing Count, Missing %, and Data Type.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    total_rows = len(df)
    missing_series = df.isna().sum()
    pct_series = (missing_series / total_rows * 100).round(2) if total_rows > 0 else 0.0

    result = pd.DataFrame({
        "Column": missing_series.index,
        "Missing Count": missing_series.values,
        "Missing %": pct_series.values,
        "Data Type": [str(df[col].dtype) for col in missing_series.index],
    })
    return result[result["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False).reset_index(drop=True)


def drop_missing_values(
    df: pd.DataFrame,
    how: str = "any",
    subset: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Drops rows containing missing values based on 'how' ('any' or 'all')
    and optional column subset.
    Returns (cleaned_df, rows_dropped_count).
    """
    if df is None or df.empty:
        return df, 0

    valid_subset = [c for c in subset if c in df.columns] if subset else None
    before_count = len(df)
    cleaned_df = df.dropna(how=how, subset=valid_subset).copy()
    rows_dropped = before_count - len(cleaned_df)
    return cleaned_df, rows_dropped


def drop_missing_columns(
    df: pd.DataFrame,
    threshold_pct: float = 50.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drops columns where missing percentage exceeds threshold_pct.
    Returns (cleaned_df, list_of_dropped_columns).
    """
    if df is None or df.empty:
        return df, []

    total_rows = len(df)
    if total_rows == 0:
        return df, []

    dropped_cols = []
    cleaned_df = df.copy()

    for col in df.columns:
        pct_missing = (cleaned_df[col].isna().sum() / total_rows) * 100
        if pct_missing >= threshold_pct:
            dropped_cols.append(col)

    if dropped_cols:
        cleaned_df = cleaned_df.drop(columns=dropped_cols)

    return cleaned_df, dropped_cols


def fill_missing_values(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    custom_value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputes missing values in specified columns.
    Supported strategies:
    - 'mean' (numeric only)
    - 'median' (numeric only)
    - 'mode' (first mode for numeric or categorical)
    - 'constant' (uses custom_value)
    - 'ffill' (forward fill)
    - 'bfill' (backward fill)

    Returns (imputed_df, imputation_log).
    """
    if df is None or df.empty or not columns:
        return df, {}

    cleaned_df = df.copy()
    log: Dict[str, Any] = {}

    for col in columns:
        if col not in cleaned_df.columns:
            continue

        missing_count = int(cleaned_df[col].isna().sum())
        if missing_count == 0:
            continue

        if strategy == "mean":
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                fill_val = cleaned_df[col].mean()
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                log[col] = {"strategy": "mean", "filled_value": round(float(fill_val), 3), "count": missing_count}
        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                fill_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                log[col] = {"strategy": "median", "filled_value": round(float(fill_val), 3), "count": missing_count}
        elif strategy == "mode":
            modes = cleaned_df[col].mode()
            if not modes.empty:
                fill_val = modes.iloc[0]
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                log[col] = {"strategy": "mode", "filled_value": str(fill_val), "count": missing_count}
        elif strategy == "constant":
            cleaned_df[col] = cleaned_df[col].fillna(custom_value)
            log[col] = {"strategy": "constant", "filled_value": str(custom_value), "count": missing_count}
        elif strategy == "ffill":
            cleaned_df[col] = cleaned_df[col].ffill()
            log[col] = {"strategy": "forward_fill", "count": missing_count}
        elif strategy == "bfill":
            cleaned_df[col] = cleaned_df[col].bfill()
            log[col] = {"strategy": "backward_fill", "count": missing_count}

    return cleaned_df, log


def detect_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> Tuple[int, pd.DataFrame]:
    """
    Identifies duplicate rows based on subset or all columns.
    Returns (duplicate_count, duplicate_rows_df).
    """
    if df is None or df.empty:
        return 0, pd.DataFrame()

    valid_subset = [c for c in subset if c in df.columns] if subset else None
    dup_mask = df.duplicated(subset=valid_subset, keep=False)
    dup_count = int(df.duplicated(subset=valid_subset, keep="first").sum())
    duplicate_rows = df[dup_mask]
    return dup_count, duplicate_rows


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> Tuple[pd.DataFrame, int]:
    """
    Removes duplicate records from DataFrame.
    Returns (deduped_df, removed_count).
    """
    if df is None or df.empty:
        return df, 0

    valid_subset = [c for c in subset if c in df.columns] if subset else None
    before_len = len(df)
    cleaned_df = df.drop_duplicates(subset=valid_subset, keep=keep).copy()
    removed_count = before_len - len(cleaned_df)
    return cleaned_df, removed_count


def convert_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: str,
    datetime_format: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str], int]:
    """
    Safely converts a column's data type without crashing.
    Supported target types:
    - 'integer'
    - 'float'
    - 'string'
    - 'boolean'
    - 'datetime'

    Returns (new_df, error_message, coerced_null_count).
    """
    if df is None or column not in df.columns:
        return df, f"Column '{column}' not found in dataset.", 0

    cleaned_df = df.copy()
    initial_nulls = int(cleaned_df[column].isna().sum())
    new_nulls = 0

    try:
        if target_type.lower() in ["integer", "int", "int64"]:
            numeric_s = pd.to_numeric(cleaned_df[column], errors="coerce")
            # Using nullable Int64 to allow retaining NaNs if present
            cleaned_df[column] = numeric_s.astype("Int64")
            new_nulls = int(cleaned_df[column].isna().sum()) - initial_nulls

        elif target_type.lower() in ["float", "float64"]:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")
            new_nulls = int(cleaned_df[column].isna().sum()) - initial_nulls

        elif target_type.lower() in ["string", "str"]:
            cleaned_df[column] = cleaned_df[column].astype(str)

        elif target_type.lower() in ["boolean", "bool"]:
            # Handle string representations like 'true', 'false', '1', '0', 'yes', 'no'
            s_lower = cleaned_df[column].astype(str).str.strip().str.lower()
            bool_map = {
                "true": True, "1": True, "yes": True, "y": True, "t": True,
                "false": False, "0": False, "no": False, "n": False, "f": False,
            }
            cleaned_df[column] = s_lower.map(bool_map)
            new_nulls = int(cleaned_df[column].isna().sum()) - initial_nulls

        elif target_type.lower() in ["datetime", "date"]:
            if datetime_format:
                cleaned_df[column] = pd.to_datetime(cleaned_df[column], format=datetime_format, errors="coerce")
            else:
                cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors="coerce")
            new_nulls = int(cleaned_df[column].isna().sum()) - initial_nulls

        else:
            return df, f"Unsupported target type '{target_type}'.", 0

        return cleaned_df, None, max(0, new_nulls)

    except Exception as e:
        return df, f"Type conversion failed: {str(e)}", 0


def detect_negative_values(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Identifies negative values in specified (or all) numerical columns.
    Returns summary DataFrame with Column, Negative Count, Negative %, Min Value.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    num_df = df.select_dtypes(include=[np.number])
    target_cols = [c for c in columns if c in num_df.columns] if columns else num_df.columns.tolist()

    findings = []
    for col in target_cols:
        series = num_df[col].dropna()
        neg_count = int((series < 0).sum())
        if neg_count > 0:
            total = len(series)
            pct = round((neg_count / total * 100), 2) if total > 0 else 0.0
            min_val = float(series.min())
            findings.append({
                "Column": col,
                "Negative Count": neg_count,
                "Negative %": pct,
                "Minimum Value": min_val,
            })

    return pd.DataFrame(findings)


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Identifies potential outliers using the Interquartile Range (IQR) rule:
    Lower Bound = Q1 - (factor * IQR)
    Upper Bound = Q3 + (factor * IQR)
    Returns summary DataFrame of detected outliers.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    num_df = df.select_dtypes(include=[np.number])
    target_cols = [c for c in columns if c in num_df.columns] if columns else num_df.columns.tolist()

    outlier_summary = []
    for col in target_cols:
        s = num_df[col].dropna()
        if len(s) < 4:
            continue

        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25

        if iqr == 0:
            continue

        lower_bound = q25 - (factor * iqr)
        upper_bound = q75 + (factor * iqr)

        outliers = s[(s < lower_bound) | (s > upper_bound)]
        outlier_count = len(outliers)

        if outlier_count > 0:
            outlier_pct = round((outlier_count / len(s) * 100), 2)
            outlier_summary.append({
                "Column": col,
                "Outlier Count": outlier_count,
                "Outlier %": outlier_pct,
                "Q1 (25%)": round(q25, 2),
                "Q3 (75%)": round(q75, 2),
                "IQR": round(iqr, 2),
                "Lower Bound": round(lower_bound, 2),
                "Upper Bound": round(upper_bound, 2),
                "Min Value": round(float(s.min()), 2),
                "Max Value": round(float(s.max()), 2),
            })

    return pd.DataFrame(outlier_summary)


def handle_outliers(
    df: pd.DataFrame,
    column: str,
    action: str = "clip",
    factor: float = 1.5,
) -> Tuple[pd.DataFrame, int]:
    """
    User-directed outlier remediation.
    Actions:
    - 'clip': Clips values to [lower_bound, upper_bound]
    - 'drop': Removes rows with outlier values
    Returns (cleaned_df, affected_rows_count).
    """
    if df is None or column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df, 0

    s = df[column].dropna()
    q25 = float(s.quantile(0.25))
    q75 = float(s.quantile(0.75))
    iqr = q75 - q25

    if iqr == 0:
        return df, 0

    lower_bound = q25 - (factor * iqr)
    upper_bound = q75 + (factor * iqr)

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    affected_count = int(outlier_mask.sum())

    if affected_count == 0:
        return df, 0

    cleaned_df = df.copy()
    if action == "clip":
        cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
    elif action == "drop":
        cleaned_df = cleaned_df[~outlier_mask].copy()

    return cleaned_df, affected_count


def get_cleaning_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    operation_description: str = "",
) -> Dict[str, Any]:
    """
    Generates a comparative summary before and after a cleaning action.
    """
    rows_before = len(before_df) if before_df is not None else 0
    rows_after = len(after_df) if after_df is not None else 0
    nulls_before = int(before_df.isna().sum().sum()) if before_df is not None else 0
    nulls_after = int(after_df.isna().sum().sum()) if after_df is not None else 0
    dups_before = int(before_df.duplicated().sum()) if before_df is not None else 0
    dups_after = int(after_df.duplicated().sum()) if after_df is not None else 0

    return {
        "operation": operation_description,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_difference": rows_after - rows_before,
        "nulls_before": nulls_before,
        "nulls_after": nulls_after,
        "nulls_difference": nulls_after - nulls_before,
        "duplicates_before": dups_before,
        "duplicates_after": dups_after,
        "duplicates_difference": dups_after - dups_before,
    }
