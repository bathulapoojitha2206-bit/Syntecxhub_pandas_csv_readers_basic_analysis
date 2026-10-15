"""
Unit Tests for Data Quality and Cleaning Module.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    detect_missing_values,
    drop_missing_values,
    drop_missing_columns,
    fill_missing_values,
    detect_duplicates,
    remove_duplicates,
    convert_column_type,
    detect_negative_values,
    detect_outliers_iqr,
    handle_outliers,
    get_cleaning_comparison,
)


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 4, 5],
        "category": ["A", "B", None, "A", "A", "C"],
        "price": [10.0, -5.0, 30.0, 40.0, 40.0, 500.0],  # -5 is negative, 500 is outlier
        "quantity": [2, 4, np.nan, 8, 8, 10],
        "flag": ["true", "false", "yes", "0", "0", "1"],
        "date": ["2024-01-01", "2024-01-02", "bad_date", "2024-01-04", "2024-01-04", "2024-01-05"],
    })


def test_detect_missing_values(messy_df):
    missing_df = detect_missing_values(messy_df)
    assert not missing_df.empty
    assert "quantity" in missing_df["Column"].values
    assert "category" in missing_df["Column"].values


def test_drop_missing_values(messy_df):
    cleaned, dropped = drop_missing_values(messy_df, how="any")
    assert dropped > 0
    assert cleaned.isna().sum().sum() == 0


def test_drop_missing_values_subset(messy_df):
    # Only drop if quantity is null
    cleaned, dropped = drop_missing_values(messy_df, subset=["quantity"])
    assert dropped == 1
    assert cleaned["quantity"].isna().sum() == 0


def test_drop_missing_columns():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [np.nan, np.nan, np.nan, 4],  # 75% missing
        "c": [10, 20, 30, 40],
    })
    cleaned, dropped_cols = drop_missing_columns(df, threshold_pct=50.0)
    assert "b" in dropped_cols
    assert "b" not in cleaned.columns
    assert "a" in cleaned.columns


def test_fill_missing_numeric_mean_median(messy_df):
    cleaned_mean, log_mean = fill_missing_values(messy_df, columns=["quantity"], strategy="mean")
    assert cleaned_mean["quantity"].isna().sum() == 0
    assert "quantity" in log_mean

    cleaned_med, log_med = fill_missing_values(messy_df, columns=["quantity"], strategy="median")
    assert cleaned_med["quantity"].isna().sum() == 0


def test_fill_missing_categorical_mode_constant(messy_df):
    cleaned_mode, log_mode = fill_missing_values(messy_df, columns=["category"], strategy="mode")
    assert cleaned_mode["category"].isna().sum() == 0
    assert cleaned_mode.loc[2, "category"] == "A"

    cleaned_const, log_const = fill_missing_values(messy_df, columns=["category"], strategy="constant", custom_value="Unknown")
    assert cleaned_const.loc[2, "category"] == "Unknown"


def test_detect_and_remove_duplicates(messy_df):
    count, dup_rows = detect_duplicates(messy_df)
    assert count == 1  # Row 4 is a duplicate of Row 3
    assert len(dup_rows) == 2

    deduped, removed = remove_duplicates(messy_df)
    assert removed == 1
    assert len(deduped) == len(messy_df) - 1


def test_convert_column_types(messy_df):
    # Convert string flag to boolean
    df_bool, err_bool, _ = convert_column_type(messy_df, "flag", "boolean")
    assert err_bool is None
    assert df_bool["flag"].dtype == bool or pd.api.types.is_bool_dtype(df_bool["flag"])

    # Convert date with bad date coerced
    df_date, err_date, coerced = convert_column_type(messy_df, "date", "datetime")
    assert err_date is None
    assert coerced >= 1
    assert pd.api.types.is_datetime64_any_dtype(df_date["date"])


def test_detect_negative_values(messy_df):
    neg_df = detect_negative_values(messy_df)
    assert not neg_df.empty
    assert "price" in neg_df["Column"].values
    assert neg_df.loc[neg_df["Column"] == "price", "Minimum Value"].iloc[0] == -5.0


def test_detect_outliers_iqr(messy_df):
    outliers = detect_outliers_iqr(messy_df, columns=["price"], factor=1.5)
    assert not outliers.empty
    assert "price" in outliers["Column"].values
    assert outliers.loc[outliers["Column"] == "price", "Outlier Count"].iloc[0] >= 1


def test_handle_outliers(messy_df):
    # Clip outlier
    clipped_df, affected = handle_outliers(messy_df, column="price", action="clip", factor=1.5)
    assert affected >= 1
    assert clipped_df["price"].max() < 500.0

    # Drop outlier
    dropped_df, affected_drop = handle_outliers(messy_df, column="price", action="drop", factor=1.5)
    assert affected_drop >= 1
    assert len(dropped_df) < len(messy_df)


def test_get_cleaning_comparison(messy_df):
    cleaned, _ = drop_missing_values(messy_df)
    comp = get_cleaning_comparison(messy_df, cleaned, "Drop Nulls")
    assert comp["rows_before"] == len(messy_df)
    assert comp["rows_after"] == len(cleaned)
    assert comp["rows_difference"] < 0
