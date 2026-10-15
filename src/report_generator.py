"""
Automated Insights and Export Report Generator Module.
Dynamically derives analytical findings, computes statistical highlights,
structures findings for CSV export, and produces comprehensive Markdown summaries.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def generate_automated_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Computes dynamic, evidence-backed analytical findings directly from the dataset.
    Never relies on hardcoded assumptions.
    """
    if df is None or df.empty:
        return [{
            "category": "Dataset Status",
            "title": "Empty Dataset",
            "finding": "The current dataset contains no records to analyze.",
            "metric_value": "0 rows",
            "severity": "Warning",
        }]

    insights: List[Dict[str, Any]] = []
    total_rows = len(df)
    total_cols = len(df.columns)

    # 1. Dataset Dimensions & Completeness
    total_cells = total_rows * total_cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0

    insights.append({
        "category": "Dimensions & Completeness",
        "title": "Dataset Volume & Density",
        "finding": f"The dataset contains {total_rows:,} rows across {total_cols} columns ({total_cells:,} total data cells). Cell fill completeness is {100 - missing_pct:.1f}%.",
        "metric_value": f"{total_rows:,} rows × {total_cols} cols",
        "severity": "Info",
    })

    # 2. Missing Values Analysis
    missing_by_col = df.isna().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0]
    if not cols_with_missing.empty:
        worst_col = cols_with_missing.idxmax()
        worst_count = int(cols_with_missing.max())
        worst_pct = round(worst_count / total_rows * 100, 2)
        insights.append({
            "category": "Data Quality",
            "title": "Missing Values Identified",
            "finding": f"{len(cols_with_missing)} of {total_cols} columns contain missing values. The column '{worst_col}' has the highest missing rate with {worst_count:,} missing values ({worst_pct}%).",
            "metric_value": f"{worst_pct}% null in '{worst_col}'",
            "severity": "Warning" if worst_pct > 15 else "Info",
        })
    else:
        insights.append({
            "category": "Data Quality",
            "title": "Clean Data Completeness",
            "finding": "Zero missing values detected across all columns. The dataset is 100% complete.",
            "metric_value": "0 missing cells",
            "severity": "Success",
        })

    # 3. Duplicate Records Analysis
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round(dup_count / total_rows * 100, 2)
        insights.append({
            "category": "Data Quality",
            "title": "Duplicate Rows Present",
            "finding": f"Detected {dup_count:,} identical duplicate rows ({dup_pct}% of total records). Consider running deduplication.",
            "metric_value": f"{dup_count} duplicates ({dup_pct}%)",
            "severity": "Warning",
        })
    else:
        insights.append({
            "category": "Data Quality",
            "title": "Unique Records",
            "finding": "Every row in the current dataset is distinct. No exact duplicate records found.",
            "metric_value": "0 duplicate rows",
            "severity": "Success",
        })

    # 4. Numerical Analysis: Highest & Lowest Averages
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        means = num_df.mean()
        if not means.empty:
            highest_avg_col = means.idxmax()
            highest_avg_val = round(float(means.max()), 2)
            lowest_avg_col = means.idxmin()
            lowest_avg_val = round(float(means.min()), 2)

            insights.append({
                "category": "Numerical Distribution",
                "title": "Highest & Lowest Metric Averages",
                "finding": f"Among {len(num_df.columns)} numeric variables, '{highest_avg_col}' holds the highest mean ({highest_avg_val:,}), while '{lowest_avg_col}' has the lowest mean ({lowest_avg_val:,}).",
                "metric_value": f"Max Avg: {highest_avg_col} ({highest_avg_val})",
                "severity": "Info",
            })

        # Outliers check
        outlier_cols_found = []
        for col in num_df.columns:
            s = num_df[col].dropna()
            if len(s) >= 4:
                q25, q75 = s.quantile(0.25), s.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    outliers = s[(s < q25 - 1.5 * iqr) | (s > q75 + 1.5 * iqr)]
                    if len(outliers) > 0:
                        outlier_cols_found.append(f"'{col}' ({len(outliers)} outliers)")

        if outlier_cols_found:
            insights.append({
                "category": "Outlier Detection",
                "title": "Statistical Outliers Flagged",
                "finding": f"IQR method identified potential outliers in: {', '.join(outlier_cols_found[:3])}{' and others' if len(outlier_cols_found) > 3 else ''}.",
                "metric_value": f"{len(outlier_cols_found)} columns with outliers",
                "severity": "Warning",
            })

    # 5. Categorical Dominance
    cat_df = df.select_dtypes(exclude=[np.number, "datetime"])
    if not cat_df.empty:
        first_cat = cat_df.columns[0]
        vc = df[first_cat].dropna().value_counts()
        if not vc.empty:
            top_cat_name = vc.index[0]
            top_cat_count = int(vc.iloc[0])
            top_cat_pct = round(top_cat_count / len(df[first_cat].dropna()) * 100, 2)

            insights.append({
                "category": "Categorical Distribution",
                "title": f"Primary Category Dominance in '{first_cat}'",
                "finding": f"In column '{first_cat}', the top category is '{top_cat_name}' accounting for {top_cat_count:,} occurrences ({top_cat_pct}% of non-null records).",
                "metric_value": f"'{top_cat_name}' ({top_cat_pct}%)",
                "severity": "Info",
            })

    # 6. Group Performance (if both numeric and categorical exist)
    if not cat_df.empty and not num_df.empty:
        cat_col = cat_df.columns[0]
        num_col = num_df.columns[0]
        # Prefer revenue, marks, or sales if present
        for nc in num_df.columns:
            if any(k in nc.lower() for k in ["revenue", "sales", "marks", "score", "attendance", "amount"]):
                num_col = nc
                break

        for cc in cat_df.columns:
            if any(k in cc.lower() for k in ["region", "category", "product", "subject", "section"]):
                cat_col = cc
                break

        grp = df.groupby(cat_col)[num_col].agg(["sum", "mean", "count"]).dropna()
        if not grp.empty:
            top_group_by_sum = grp["sum"].idxmax()
            top_sum_val = round(float(grp["sum"].max()), 2)
            top_group_by_avg = grp["mean"].idxmax()
            top_avg_val = round(float(grp["mean"].max()), 2)

            insights.append({
                "category": "Segment Performance",
                "title": f"Top Segment in '{cat_col}' by '{num_col}'",
                "finding": f"'{top_group_by_sum}' leads '{cat_col}' in aggregate {num_col} with a total of {top_sum_val:,}. The highest per-unit average is led by '{top_group_by_avg}' (mean: {top_avg_val:,}).",
                "metric_value": f"Top: {top_group_by_sum} ({top_sum_val:,})",
                "severity": "Success",
            })

    # 7. Correlation Highlights
    if num_df.shape[1] >= 2:
        corr_matrix = num_df.corr().abs()
        for c in corr_matrix.columns:
            corr_matrix.loc[c, c] = 0.0
        max_corr_val = float(corr_matrix.max().max())

        if not np.isnan(max_corr_val) and max_corr_val > 0.01:
            # Find the pair
            unstacked = corr_matrix.unstack().sort_values(ascending=False)
            pair = unstacked.index[0]
            col1, col2 = pair
            actual_corr = round(float(num_df[col1].corr(num_df[col2])), 3)

            corr_direction = "positive" if actual_corr > 0 else "negative"
            strength = "strong" if abs(actual_corr) >= 0.7 else ("moderate" if abs(actual_corr) >= 0.4 else "weak")

            insights.append({
                "category": "Correlation Analysis",
                "title": f"Strongest Metric Relationship: '{col1}' & '{col2}'",
                "finding": f"A {strength} {corr_direction} correlation (r = {actual_corr}) exists between '{col1}' and '{col2}'. Note: Correlation does not imply causation.",
                "metric_value": f"r = {actual_corr} ({strength})",
                "severity": "Info",
            })

    # 8. Date / Temporal Range Analysis
    date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    parsed_date_col = None
    for cand in date_candidates:
        dt = pd.to_datetime(df[cand], errors="coerce")
        if dt.notna().sum() > len(df) * 0.7:
            parsed_date_col = cand
            break

    if parsed_date_col:
        dt_series = pd.to_datetime(df[parsed_date_col], errors="coerce").dropna()
        earliest = dt_series.min().strftime("%Y-%m-%d")
        latest = dt_series.max().strftime("%Y-%m-%d")
        span_days = (dt_series.max() - dt_series.min()).days

        insights.append({
            "category": "Temporal Scope",
            "title": f"Date Range in '{parsed_date_col}'",
            "finding": f"Records span from {earliest} to {latest} (covering {span_days:,} days).",
            "metric_value": f"{earliest} to {latest}",
            "severity": "Info",
        })

    return insights


def generate_insights_dataframe(insights: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts automated insights list into a clean tabular DataFrame for display and CSV export."""
    if not insights:
        return pd.DataFrame()
    return pd.DataFrame(insights)


def generate_markdown_report(
    df: pd.DataFrame,
    insights: List[Dict[str, Any]],
    cleaning_history: Optional[List[Dict[str, Any]]] = None,
    dataset_name: str = "Uploaded Dataset",
) -> str:
    """
    Constructs a comprehensive, polished Markdown analysis report.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_rows = len(df) if df is not None else 0
    total_cols = len(df.columns) if df is not None else 0

    lines = [
        f"# Automated Data Analysis Report: {dataset_name}",
        f"*Generated on: {now_str}*",
        "",
        "## 1. Executive Summary",
        f"- **Dataset Name:** {dataset_name}",
        f"- **Current Record Count:** {total_rows:,} rows",
        f"- **Current Feature Count:** {total_cols} columns",
        f"- **Total Missing Cells:** {int(df.isna().sum().sum()):,} ({round(df.isna().sum().sum() / (total_rows * total_cols) * 100, 2) if total_rows * total_cols > 0 else 0}%)",
        f"- **Duplicate Rows:** {int(df.duplicated().sum()):,}",
        "",
        "## 2. Dynamic Key Findings",
    ]

    for idx, item in enumerate(insights, 1):
        lines.append(f"### {idx}. {item.get('title', 'Insight')}")
        lines.append(f"- **Category:** {item.get('category', 'General')}")
        lines.append(f"- **Metric Value:** `{item.get('metric_value', 'N/A')}`")
        lines.append(f"- **Finding:** {item.get('finding', '')}")
        lines.append("")

    if cleaning_history:
        lines.append("## 3. Data Cleaning Operations Audit Trail")
        lines.append("| Step | Operation | Rows Affected | Details |")
        lines.append("|---|---|---|---|")
        for i, step in enumerate(cleaning_history, 1):
            op = step.get("operation", "Cleaning")
            diff = step.get("rows_difference", 0)
            det = step.get("details", "")
            lines.append(f"| {i} | {op} | {diff:+d} rows | {det} |")
        lines.append("")

    lines.append("## 4. Analytical Methodology & Disclaimer")
    lines.append("- All metrics and statistics were computed dynamically using Pandas vectorized routines.")
    lines.append("- **Important Statistical Note:** Correlation metrics represent linear associations and do not prove causal relationships.")
    lines.append("")

    return "\n".join(lines)
