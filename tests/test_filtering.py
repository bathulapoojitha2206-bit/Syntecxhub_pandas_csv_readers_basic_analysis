"""
Unit Tests for Filtering, Slicing, and Sorting Module.
"""

import pandas as pd
import pytest

from src.filtering import (
    filter_numeric_range,
    filter_categorical_isin,
    filter_date_range,
    filter_text_search,
    slice_loc,
    slice_iloc,
    apply_query_filter,
    sort_dataframe,
    select_columns,
    drop_columns,
    rename_columns,
    generate_pandas_filter_code,
)


@pytest.fixture
def sample_filter_df():
    return pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4", "ORD-5"],
        "product": ["Laptop Pro", "Wireless Mouse", "Office Desk", "Gaming Mouse", "Laptop Air"],
        "category": ["Electronics", "Electronics", "Furniture", "Electronics", "Electronics"],
        "price": [1200.0, 30.0, 350.0, 80.0, 950.0],
        "quantity": [1, 5, 2, 3, 2],
        "order_date": ["2024-01-10", "2024-01-15", "2024-02-01", "2024-02-10", "2024-03-05"],
    })


def test_filter_numeric_range(sample_filter_df):
    res = filter_numeric_range(sample_filter_df, "price", 50.0, 1000.0)
    assert len(res) == 3
    assert set(res["order_id"]) == {"ORD-3", "ORD-4", "ORD-5"}


def test_filter_categorical_isin(sample_filter_df):
    res = filter_categorical_isin(sample_filter_df, "category", ["Furniture"])
    assert len(res) == 1
    assert res.iloc[0]["order_id"] == "ORD-3"


def test_filter_date_range(sample_filter_df):
    res = filter_date_range(sample_filter_df, "order_date", "2024-02-01", "2024-02-28")
    assert len(res) == 2
    assert set(res["order_id"]) == {"ORD-3", "ORD-4"}


def test_filter_text_search(sample_filter_df):
    res = filter_text_search(sample_filter_df, "product", "mouse", case_sensitive=False)
    assert len(res) == 2
    assert set(res["order_id"]) == {"ORD-2", "ORD-4"}


def test_slice_loc(sample_filter_df):
    res = slice_loc(
        sample_filter_df,
        row_condition_col="quantity",
        row_condition_op=">=",
        row_condition_val=3,
        columns=["order_id", "product", "quantity"],
    )
    assert len(res) == 2
    assert list(res.columns) == ["order_id", "product", "quantity"]


def test_slice_iloc(sample_filter_df):
    res = slice_iloc(sample_filter_df, row_start=1, row_end=3, col_start=0, col_end=2)
    assert res.shape == (2, 2)
    assert list(res.columns) == ["order_id", "product"]


def test_apply_query_filter(sample_filter_df):
    res, err = apply_query_filter(sample_filter_df, "price > 100 and quantity >= 2")
    assert err is None
    assert len(res) == 2  # ORD-3 ($350, 2) and ORD-5 ($950, 2)

    # Disallowed tokens check
    res_bad, err_bad = apply_query_filter(sample_filter_df, "__import__('os').system('ls')")
    assert err_bad is not None
    assert "disallowed" in err_bad.lower()


def test_sort_dataframe(sample_filter_df):
    res = sort_dataframe(sample_filter_df, by_columns=["price"], ascending=False)
    assert res.iloc[0]["order_id"] == "ORD-1"  # Max price 1200.0


def test_column_manipulation(sample_filter_df):
    sel = select_columns(sample_filter_df, ["order_id", "price"])
    assert list(sel.columns) == ["order_id", "price"]

    dropped = drop_columns(sample_filter_df, ["order_date"])
    assert "order_date" not in dropped.columns

    renamed = rename_columns(sample_filter_df, {"order_id": "ID", "price": "Unit_Cost"})
    assert "ID" in renamed.columns
    assert "Unit_Cost" in renamed.columns


def test_generate_pandas_filter_code():
    code = generate_pandas_filter_code(
        numeric_filters=[{"column": "price", "min": 10, "max": 100}],
        cat_filters=[{"column": "category", "values": ["Electronics"]}],
        sort_cols=["price"],
        sort_ascending=True,
        selected_cols=["order_id", "price"],
    )
    assert "filtered_df" in code
    assert "price" in code
    assert "Electronics" in code
    assert "sort_values" in code
