"""
Interactive Data Visualization Module.
Constructs Plotly charts for bar charts, line graphs, pie/donut charts,
histograms, scatter plots, box plots, missing value distributions, and correlation heatmaps.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Consistent modern color sequence
COLOR_PALETTE = [
    "#4F46E5",  # Indigo
    "#06B6D4",  # Cyan
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#EC4899",  # Pink
    "#8B5CF6",  # Violet
    "#3B82F6",  # Blue
    "#EF4444",  # Red
]


def _apply_theme(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    """Applies clean, modern styling to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        title={
            "text": title or "",
            "font": {"size": 16, "family": "Inter, sans-serif", "color": "#1F2937"},
            "x": 0.02,
            "xanchor": "left",
        },
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        font={"family": "Inter, sans-serif", "color": "#374151"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hoverlabel={"font_family": "Inter, sans-serif"},
    )
    return fig


def plot_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    barmode: str = "group",
    orientation: str = "v",
    title: Optional[str] = None,
) -> go.Figure:
    """Creates an interactive bar chart."""
    if df is None or df.empty or x_col not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No data available for Bar Chart")

    if y_col and y_col in df.columns:
        fig = px.bar(
            df,
            x=x_col if orientation == "v" else y_col,
            y=y_col if orientation == "v" else x_col,
            color=color_col if color_col in df.columns else None,
            barmode=barmode,
            orientation=orientation,
            color_discrete_sequence=COLOR_PALETTE,
        )
    else:
        # Frequency count of x_col
        counts = df[x_col].value_counts().reset_index()
        counts.columns = [x_col, "Count"]
        fig = px.bar(
            counts,
            x=x_col,
            y="Count",
            color_discrete_sequence=COLOR_PALETTE,
        )

    title_text = title or (f"{y_col} by {x_col}" if y_col else f"Distribution of {x_col}")
    return _apply_theme(fig, title_text)


def plot_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    markers: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """Creates an interactive line chart."""
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No data available for Line Chart")

    sorted_df = df.sort_values(by=x_col)
    fig = px.line(
        sorted_df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in df.columns else None,
        markers=markers,
        color_discrete_sequence=COLOR_PALETTE,
    )
    title_text = title or f"{y_col} over {x_col}"
    return _apply_theme(fig, title_text)


def plot_pie_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: Optional[str] = None,
    hole: float = 0.4,
    title: Optional[str] = None,
) -> go.Figure:
    """Creates an interactive pie / donut chart."""
    if df is None or df.empty or names_col not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No data available for Pie Chart")

    if values_col and values_col in df.columns:
        grouped = df.groupby(names_col)[values_col].sum().reset_index()
        fig = px.pie(
            grouped,
            names=names_col,
            values=values_col,
            hole=hole,
            color_discrete_sequence=COLOR_PALETTE,
        )
    else:
        counts = df[names_col].value_counts().reset_index()
        counts.columns = [names_col, "Count"]
        fig = px.pie(
            counts,
            names=names_col,
            values="Count",
            hole=hole,
            color_discrete_sequence=COLOR_PALETTE,
        )

    title_text = title or f"Proportions of {names_col}"
    return _apply_theme(fig, title_text)


def plot_histogram(
    df: pd.DataFrame,
    x_col: str,
    nbins: int = 30,
    color_col: Optional[str] = None,
    marginal: Optional[str] = "box",
    title: Optional[str] = None,
) -> go.Figure:
    """Creates an interactive histogram with optional marginal distribution."""
    if df is None or df.empty or x_col not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No data available for Histogram")

    fig = px.histogram(
        df,
        x=x_col,
        nbins=nbins,
        color=color_col if color_col in df.columns else None,
        marginal=marginal,
        color_discrete_sequence=COLOR_PALETTE,
    )
    title_text = title or f"Histogram Distribution of {x_col}"
    return _apply_theme(fig, title_text)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    trendline: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Creates an interactive scatter plot with optional trendline."""
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No data available for Scatter Plot")

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in df.columns else None,
        size=size_col if size_col in df.columns else None,
        trendline=trendline,
        color_discrete_sequence=COLOR_PALETTE,
    )
    title_text = title or f"{y_col} vs {x_col}"
    return _apply_theme(fig, title_text)


def plot_box(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    color_col: Optional[str] = None,
    points: str = "outliers",
    title: Optional[str] = None,
) -> go.Figure:
    """Creates an interactive box plot."""
    if df is None or df.empty or y_col not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No data available for Box Plot")

    fig = px.box(
        df,
        y=y_col,
        x=x_col if x_col in df.columns else None,
        color=color_col if color_col in df.columns else None,
        points=points,
        color_discrete_sequence=COLOR_PALETTE,
    )
    title_text = title or (f"Box Plot of {y_col} by {x_col}" if x_col else f"Box Plot of {y_col}")
    return _apply_theme(fig, title_text)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Creates an interactive correlation heatmap for numeric columns.
    """
    if df is None or df.empty:
        fig = go.Figure()
        return _apply_theme(fig, "No data available for Correlation Heatmap")

    num_df = df.select_dtypes(include=[np.number])
    if numeric_cols:
        valid_cols = [c for c in numeric_cols if c in num_df.columns]
        num_df = num_df[valid_cols]

    if num_df.shape[1] < 2:
        fig = go.Figure()
        return _apply_theme(fig, "Need at least 2 numeric columns for correlation heatmap")

    corr = num_df.corr().round(3)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    title_text = title or "Pearson Correlation Matrix"
    return _apply_theme(fig, title_text)


def plot_missing_values_bar(df: pd.DataFrame) -> go.Figure:
    """
    Visualizes missing value counts and percentages across all columns.
    """
    if df is None or df.empty:
        fig = go.Figure()
        return _apply_theme(fig, "No missing values to display")

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(1)

    chart_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values,
    }).sort_values(by="Missing Count", ascending=True)

    fig = px.bar(
        chart_df,
        x="Missing Count",
        y="Column",
        orientation="h",
        text="Missing %",
        color="Missing %",
        color_continuous_scale="Reds",
        title="Missing Values per Column",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    return _apply_theme(fig, "Missing Values per Column (Count and %)")
