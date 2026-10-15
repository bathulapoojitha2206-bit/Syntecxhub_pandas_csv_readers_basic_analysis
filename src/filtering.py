"""
Filtering, Slicing, and Sorting Module.
Implements Pandas-idiomatic filtering (loc, iloc, query, isin, between),
sorting, column renaming/dropping, and automated Pandas code snippet generation.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def filter_numeric_range(
    df: pd.DataFrame,
    column: str,
    min_val: float,
    max_val: float,
) -> pd.DataFrame:
    """Filters DataFrame rows where column values fall between min_val and max_val inclusive."""
    if df is None or column not in df.columns:
        return df
    return df[df[column].between(min_val, max_val, inclusive="both")]


def filter_categorical_isin(
    df: pd.DataFrame,
    column: str,
    selected_values: List[Any],
) -> pd.DataFrame:
    """Filters DataFrame rows where column values match any in selected_values using isin()."""
    if df is None or column not in df.columns or not selected_values:
        return df
    # Match both original and string representations for flexibility
    return df[df[column].isin(selected_values) | df[column].astype(str).isin([str(v) for v in selected_values])]


def filter_date_range(
    df: pd.DataFrame,
    column: str,
    start_date: Union[date, datetime, str],
    end_date: Union[date, datetime, str],
) -> pd.DataFrame:
    """Filters DataFrame rows where a datetime-parseable column falls within start_date and end_date."""
    if df is None or column not in df.columns:
        return df

    dt_series = pd.to_datetime(df[column], errors="coerce")
    ts_start = pd.to_datetime(start_date)
    ts_end = pd.to_datetime(end_date)
    mask = (dt_series >= ts_start) & (dt_series <= ts_end)
    return df[mask]


def filter_text_search(
    df: pd.DataFrame,
    column: str,
    search_query: str,
    case_sensitive: bool = False,
    regex: bool = False,
) -> pd.DataFrame:
    """Filters rows where the column string contains search_query using str.contains()."""
    if df is None or column not in df.columns or not search_query.strip():
        return df

    mask = df[column].astype(str).str.contains(
        search_query,
        case=case_sensitive,
        regex=regex,
        na=False,
    )
    return df[mask]


def slice_loc(
    df: pd.DataFrame,
    row_condition_col: Optional[str] = None,
    row_condition_op: str = "==",
    row_condition_val: Any = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Demonstrates Pandas .loc[] row and column label slicing.
    """
    if df is None or df.empty:
        return df

    selected_cols = [c for c in columns if c in df.columns] if columns else slice(None)

    if not row_condition_col or row_condition_col not in df.columns:
        return df.loc[:, selected_cols]

    col = df[row_condition_col]
    if row_condition_op == "==":
        mask = col == row_condition_val
    elif row_condition_op == "!=":
        mask = col != row_condition_val
    elif row_condition_op == ">":
        mask = col > float(row_condition_val)
    elif row_condition_op == ">=":
        mask = col >= float(row_condition_val)
    elif row_condition_op == "<":
        mask = col < float(row_condition_val)
    elif row_condition_op == "<=":
        mask = col <= float(row_condition_val)
    else:
        mask = pd.Series(True, index=df.index)

    return df.loc[mask, selected_cols]


def slice_iloc(
    df: pd.DataFrame,
    row_start: int = 0,
    row_end: Optional[int] = None,
    col_start: int = 0,
    col_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Demonstrates Pandas .iloc[] integer position-based slicing.
    """
    if df is None or df.empty:
        return df

    r_end = row_end if row_end is not None else len(df)
    c_end = col_end if col_end is not None else len(df.columns)
    return df.iloc[row_start:r_end, col_start:c_end]


def apply_query_filter(
    df: pd.DataFrame,
    query_string: str,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Safely applies a Pandas df.query() string.
    Returns (filtered_df, error_message).
    """
    if df is None or df.empty or not query_string.strip():
        return df, None

    # Check that query string doesn't contain unsafe calls
    forbidden_tokens = ["import", "eval", "exec", "__", "os.", "sys.", "open("]
    for token in forbidden_tokens:
        if token in query_string.lower():
            return df, f"Disallowed token '{token}' in query."

    try:
        filtered = df.query(query_string)
        return filtered, None
    except Exception as e:
        return df, f"Query error: {str(e)}"


def sort_dataframe(
    df: pd.DataFrame,
    by_columns: List[str],
    ascending: Union[bool, List[bool]] = True,
) -> pd.DataFrame:
    """Sorts DataFrame values by specified columns."""
    if df is None or not by_columns:
        return df

    valid_cols = [c for c in by_columns if c in df.columns]
    if not valid_cols:
        return df

    return df.sort_values(by=valid_cols, ascending=ascending)


def select_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """Subsets DataFrame to specified columns."""
    if df is None or not columns:
        return df
    valid_cols = [c for c in columns if c in df.columns]
    return df[valid_cols] if valid_cols else df


def drop_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """Drops specified columns using drop(columns=...)."""
    if df is None or not columns:
        return df
    valid_cols = [c for c in columns if c in df.columns]
    return df.drop(columns=valid_cols)


def rename_columns(
    df: pd.DataFrame,
    rename_mapping: Dict[str, str],
) -> pd.DataFrame:
    """Renames columns using rename(columns=...)."""
    if df is None or not rename_mapping:
        return df
    return df.rename(columns=rename_mapping)


def generate_pandas_filter_code(
    numeric_filters: Optional[List[Dict[str, Any]]] = None,
    cat_filters: Optional[List[Dict[str, Any]]] = None,
    sort_cols: Optional[List[str]] = None,
    sort_ascending: bool = True,
    selected_cols: Optional[List[str]] = None,
) -> str:
    """
    Constructs clean, reproducible Pandas Python code representing applied filters.
    """
    lines = ["# Reproducible Pandas Code for applied filters", "filtered_df = df.copy()"]

    if numeric_filters:
        for nf in numeric_filters:
            col, mn, mx = nf["column"], nf["min"], nf["max"]
            lines.append(f"filtered_df = filtered_df[filtered_df['{col}'].between({mn}, {mx})]")

    if cat_filters:
        for cf in cat_filters:
            col, vals = cf["column"], cf["values"]
            lines.append(f"filtered_df = filtered_df[filtered_df['{col}'].isin({vals})]")

    if sort_cols:
        lines.append(f"filtered_df = filtered_df.sort_values(by={sort_cols}, ascending={sort_ascending})")

    if selected_cols:
        lines.append(f"filtered_df = filtered_df[{selected_cols}]")

    return "\n".join(lines)
