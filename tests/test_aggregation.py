"""
Unit Tests for Grouping and Aggregation Module.
"""

import pandas as pd
import pytest

from src.aggregation import (
    perform_groupby,
    create_pivot_table,
    create_crosstab,
    get_dynamic_aggregation_suggestions,
)


@pytest.fixture
def sales_agg_df():
    return pd.DataFrame({
        "region": ["North", "North", "South", "South", "West", "West"],
        "category": ["Electronics", "Furniture", "Electronics", "Furniture", "Electronics", "Furniture"],
        "revenue": [500.0, 300.0, 700.0, 450.0, 600.0, 800.0],
        "quantity": [5, 2, 8, 3, 6, 4],
    })


def test_perform_groupby_single_group(sales_agg_df):
    res = perform_groupby(
        sales_agg_df,
        group_columns=["region"],
        target_columns=["revenue"],
        agg_functions=["sum", "mean"],
    )
    assert not res.empty
    assert "region" in res.columns
    assert "revenue_sum" in res.columns
    assert "revenue_mean" in res.columns
    assert len(res) == 3


def test_perform_groupby_multi_group(sales_agg_df):
    res = perform_groupby(
        sales_agg_df,
        group_columns=["region", "category"],
        target_columns=["revenue", "quantity"],
        agg_functions=["sum"],
    )
    assert not res.empty
    assert len(res) == 6
    assert "revenue_sum" in res.columns
    assert "quantity_sum" in res.columns


def test_create_pivot_table(sales_agg_df):
    pivot = create_pivot_table(
        sales_agg_df,
        index_col="region",
        columns_col="category",
        values_col="revenue",
        aggfunc="sum",
    )
    assert not pivot.empty
    assert "Electronics" in pivot.columns
    assert "Furniture" in pivot.columns
    assert pivot.loc["North", "Electronics"] == 500.0


def test_create_crosstab(sales_agg_df):
    ct = create_crosstab(sales_agg_df, row_col="region", col_col="category")
    assert not ct.empty
    assert ct.shape == (3, 2)
    assert ct.loc["North", "Electronics"] == 1

    # Normalized
    ct_norm = create_crosstab(sales_agg_df, row_col="region", col_col="category", normalize="all")
    assert not ct_norm.empty


def test_dynamic_aggregation_suggestions(sales_agg_df):
    suggestions = get_dynamic_aggregation_suggestions(sales_agg_df)
    assert len(suggestions) >= 1
    first_sug = suggestions[0]
    assert "group_cols" in first_sug
    assert "target_cols" in first_sug
    assert "agg_funcs" in first_sug
    # Check that suggested columns actually exist in sales_agg_df
    assert all(c in sales_agg_df.columns for c in first_sug["group_cols"])
    assert all(c in sales_agg_df.columns for c in first_sug["target_cols"])


def test_groupby_empty_or_invalid():
    res = perform_groupby(pd.DataFrame(), ["a"], ["b"], ["sum"])
    assert res.empty

    df = pd.DataFrame({"x": [1, 2]})
    res_invalid = perform_groupby(df, ["non_existent"], ["x"], ["sum"])
    assert res_invalid.empty
