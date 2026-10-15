"""
Unit Tests for Data Overview, Reporting, and Insights Module.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_overview import (
    get_dataset_kpis,
    get_column_profile,
    get_head,
    get_tail,
    get_sample,
    get_numerical_summary,
    get_categorical_summary,
    get_column_value_counts,
)
from src.report_generator import (
    generate_automated_insights,
    generate_insights_dataframe,
    generate_markdown_report,
)


@pytest.fixture
def analytics_df():
    return pd.DataFrame({
        "student_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "subject": ["Math", "Math", "Physics", "Physics", "Chemistry", "Chemistry"],
        "marks": [85.0, 90.0, 78.0, 82.0, 88.0, 95.0],
        "attendance": [92.0, 95.0, 85.0, 88.0, 90.0, 96.0],
        "section": ["A", "A", "B", "B", "A", "A"],
    })


def test_get_dataset_kpis(analytics_df):
    kpis = get_dataset_kpis(analytics_df)
    assert kpis["total_rows"] == 6
    assert kpis["total_columns"] == 5
    assert kpis["numeric_columns_count"] == 2  # marks, attendance
    assert kpis["categorical_columns_count"] == 3  # student_id, subject, section
    assert kpis["total_missing_values"] == 0
    assert kpis["duplicate_rows_count"] == 0


def test_get_column_profile(analytics_df):
    profile = get_column_profile(analytics_df)
    assert not profile.empty
    assert len(profile) == 5
    assert "Column Name" in profile.columns
    assert "Missing Count" in profile.columns
    assert "Unique Count" in profile.columns


def test_head_tail_sample(analytics_df):
    assert len(get_head(analytics_df, 3)) == 3
    assert len(get_tail(analytics_df, 2)) == 2
    assert len(get_sample(analytics_df, 4)) == 4


def test_get_numerical_summary(analytics_df):
    summary = get_numerical_summary(analytics_df)
    assert not summary.empty
    assert "marks" in summary.columns
    assert "attendance" in summary.columns
    assert "Mean" in summary.index
    assert "Median" in summary.index
    assert "IQR" in summary.index
    assert "Skewness" in summary.index


def test_get_categorical_summary(analytics_df):
    summary = get_categorical_summary(analytics_df)
    assert not summary.empty
    assert "Column Name" in summary.columns
    assert "Most Frequent" in summary.columns


def test_get_column_value_counts(analytics_df):
    vc = get_column_value_counts(analytics_df, "section")
    assert not vc.empty
    assert "Count" in vc.columns
    assert "Percentage" in vc.columns


def test_generate_automated_insights(analytics_df):
    insights = generate_automated_insights(analytics_df)
    assert len(insights) >= 3
    categories = [item["category"] for item in insights]
    assert "Dimensions & Completeness" in categories
    assert "Numerical Distribution" in categories

    # Convert to DataFrame
    idf = generate_insights_dataframe(insights)
    assert not idf.empty
    assert "title" in idf.columns
    assert "finding" in idf.columns


def test_generate_markdown_report(analytics_df):
    insights = generate_automated_insights(analytics_df)
    md = generate_markdown_report(analytics_df, insights, dataset_name="Student Test Data")
    assert "Automated Data Analysis Report" in md
    assert "Student Test Data" in md
    assert "Executive Summary" in md
