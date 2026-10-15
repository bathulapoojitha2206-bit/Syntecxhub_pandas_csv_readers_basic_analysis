"""
Grouping, Aggregation, and Pivot Table Module.
Implements groupby, multi-metric agg, pivot_table, crosstab, value_counts,
and dynamic aggregation recipe suggestions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def perform_groupby(
    df: pd.DataFrame,
    group_columns: List[str],
    target_columns: List[str],
    agg_functions: List[str],
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Performs multi-column groupby aggregation with multiple aggregation functions.
    Flattens multi-level column names and calls reset_index().
    """
    if df is None or not group_columns or not target_columns or not agg_functions:
        return pd.DataFrame()

    valid_groups = [c for c in group_columns if c in df.columns]
    valid_targets = [c for c in target_columns if c in df.columns]

    if not valid_groups or not valid_targets:
        return pd.DataFrame()

    agg_dict = {t: agg_functions for t in valid_targets}

    try:
        grouped = df.groupby(valid_groups).agg(agg_dict)
    except Exception as e:
        # Fallback if some agg cannot be computed
        return pd.DataFrame()

    # Flatten multi-level columns if necessary
    if isinstance(grouped.columns, pd.MultiIndex):
        flat_cols = []
        for col_name, func_name in grouped.columns:
            flat_cols.append(f"{col_name}_{func_name}")
        grouped.columns = flat_cols

    result = grouped.reset_index()

    # Optional sorting
    if sort_by and sort_by in result.columns:
        result = result.sort_values(by=sort_by, ascending=ascending)

    return result


def create_pivot_table(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    values_col: str,
    aggfunc: str = "mean",
    fill_value: Union[int, float] = 0,
) -> pd.DataFrame:
    """
    Creates a pivot table using pd.pivot_table().
    """
    if df is None or df.empty:
        return pd.DataFrame()

    for c in [index_col, columns_col, values_col]:
        if c not in df.columns:
            return pd.DataFrame()

    try:
        pivot = pd.pivot_table(
            df,
            index=index_col,
            columns=columns_col,
            values=values_col,
            aggfunc=aggfunc,
            fill_value=fill_value,
        )
        return pivot
    except Exception:
        return pd.DataFrame()


def create_crosstab(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """
    Computes a frequency crosstabulation using pd.crosstab().
    normalize can be None, 'all', 'index', or 'columns'.
    """
    if df is None or row_col not in df.columns or col_col not in df.columns:
        return pd.DataFrame()

    norm_val = normalize if normalize in ["all", "index", "columns"] else False
    try:
        ct = pd.crosstab(df[row_col], df[col_col], normalize=norm_val)
        if norm_val:
            ct = (ct * 100).round(2)
        return ct
    except Exception:
        return pd.DataFrame()


def get_dynamic_aggregation_suggestions(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Inspects columns of the uploaded DataFrame and generates smart,
    contextual aggregation examples without hardcoding.
    """
    if df is None or df.empty:
        return []

    cat_cols = df.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not cat_cols or not num_cols:
        return []

    suggestions = []

    # Priority keywords detection
    priority_cat_keywords = ["region", "category", "product", "subject", "section", "department", "status"]
    priority_num_keywords = ["revenue", "sales", "marks", "attendance", "amount", "profit", "price", "score"]

    best_cat = None
    for kw in priority_cat_keywords:
        match = [c for c in cat_cols if kw in c.lower()]
        if match:
            best_cat = match[0]
            break
    if not best_cat:
        best_cat = cat_cols[0]

    best_num = None
    for kw in priority_num_keywords:
        match = [c for c in num_cols if kw in c.lower()]
        if match:
            best_num = match[0]
            break
    if not best_num:
        best_num = num_cols[0]

    # Suggestion 1: Total and Mean of primary metric by category
    suggestions.append({
        "label": f"Total & Average {best_num} by {best_cat}",
        "group_cols": [best_cat],
        "target_cols": [best_num],
        "agg_funcs": ["sum", "mean"],
        "sort_col": f"{best_num}_sum",
    })

    # Suggestion 2: If secondary categorical exists, suggest pivot table or secondary group
    other_cats = [c for c in cat_cols if c != best_cat]
    if other_cats:
        sec_cat = other_cats[0]
        suggestions.append({
            "label": f"Average {best_num} by {best_cat} and {sec_cat}",
            "group_cols": [best_cat, sec_cat],
            "target_cols": [best_num],
            "agg_funcs": ["mean", "count"],
            "sort_col": f"{best_num}_mean",
        })

    # Suggestion 3: Count of records across primary categorical
    suggestions.append({
        "label": f"Record Counts & Median {best_num} by {best_cat}",
        "group_cols": [best_cat],
        "target_cols": [best_num],
        "agg_funcs": ["count", "median", "min", "max"],
        "sort_col": f"{best_num}_count",
    })

    return suggestions
