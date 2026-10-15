"""
Pandas CSV Reader & Basic Data Analysis - Streamlit Dashboard
An interactive portfolio-grade analytical tool built with Python, Pandas, NumPy, and Plotly.
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from src.csv_reader import (
    load_csv,
    get_sample_datasets,
    get_preview_data,
    get_csv_download_buffer,
)
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
from src.aggregation import (
    perform_groupby,
    create_pivot_table,
    create_crosstab,
    get_dynamic_aggregation_suggestions,
)
from src.visualization import (
    plot_bar_chart,
    plot_line_chart,
    plot_pie_chart,
    plot_histogram,
    plot_scatter,
    plot_box,
    plot_correlation_heatmap,
    plot_missing_values_bar,
)
from src.report_generator import (
    generate_automated_insights,
    generate_insights_dataframe,
    generate_markdown_report,
)

# ---------------------------------------------------------
# Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pandas CSV Reader & Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern UI Styling
st.markdown(
    """
    <style>
    /* Global styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 4px;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Badge tags */
    .badge-tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-primary { background-color: #e0e7ff; color: #4338ca; }
    .badge-success { background-color: #d1fae5; color: #065f46; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    
    /* Code preview box */
    .code-box {
        background-color: #0f172a;
        color: #e2e8f0;
        padding: 14px 18px;
        border-radius: 8px;
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "None"
if "cleaning_history" not in st.session_state:
    st.session_state.cleaning_history = []
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None

# Automatically load sample sales dataset on initial launch if nothing is loaded
if st.session_state.df is None:
    sample_paths = get_sample_datasets()
    sales_path = sample_paths.get("Sample Sales Dataset (Synthetic)")
    if sales_path and os.path.exists(sales_path):
        loaded_df, err, meta = load_csv(sales_path)
        if loaded_df is not None:
            st.session_state.df = loaded_df.copy()
            st.session_state.raw_df = loaded_df.copy()
            st.session_state.dataset_name = "Sample Sales Dataset (Synthetic)"


# ---------------------------------------------------------
# Sidebar Navigation & Controls
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=56)
st.sidebar.title("Pandas Analytics")
st.sidebar.caption("Interactive CSV Explorer & Analyst Toolkit")

pages = [
    "1. Home & Dataset Overview",
    "2. CSV Upload & Data Preview",
    "3. Data Quality & Cleaning",
    "4. Filtering & Sorting",
    "5. Statistical Analysis",
    "6. Grouping & Aggregation",
    "7. Data Visualization",
    "8. Export & Analysis Report",
]

selected_page = st.sidebar.radio("Navigation", pages, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset In Memory")
if st.session_state.df is not None:
    st.sidebar.markdown(f"**Source:** `{st.session_state.dataset_name}`")
    st.sidebar.markdown(f"**Shape:** `{len(st.session_state.df):,} rows × {len(st.session_state.df.columns)} cols`")
    if st.sidebar.button("🔄 Reset to Original Data", use_container_width=True):
        if st.session_state.raw_df is not None:
            st.session_state.df = st.session_state.raw_df.copy()
            st.session_state.cleaning_history = []
            st.sidebar.success("Dataset reset to original state!")
            st.rerun()
else:
    st.sidebar.warning("No dataset loaded.")

# Sidebar Sample Dataset Quick Loader
st.sidebar.markdown("---")
st.sidebar.caption("Quick Switch Sample Data")
sample_dict = get_sample_datasets()
selected_sample = st.sidebar.selectbox("Choose Sample Dataset", list(sample_dict.keys()))
if st.sidebar.button("Load Selected Sample", use_container_width=True):
    path = sample_dict[selected_sample]
    s_df, s_err, s_meta = load_csv(path)
    if s_df is not None:
        st.session_state.df = s_df.copy()
        st.session_state.raw_df = s_df.copy()
        st.session_state.dataset_name = selected_sample
        st.session_state.cleaning_history = []
        st.sidebar.success(f"Loaded {selected_sample}!")
        st.rerun()
    else:
        st.sidebar.error(f"Error: {s_err}")


# Helper function to render KPI ribbons
def render_kpi_metrics(df: pd.DataFrame):
    kpis = get_dataset_kpis(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Rows</div>
                <div class="metric-value">{kpis['total_rows']:,}</div>
                <div class="metric-subtitle">Records in memory</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Columns</div>
                <div class="metric-value">{kpis['total_columns']}</div>
                <div class="metric-subtitle">{kpis['numeric_columns_count']} num | {kpis['categorical_columns_count']} cat</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Missing Cells</div>
                <div class="metric-value">{kpis['total_missing_values']:,}</div>
                <div class="metric-subtitle">{kpis['missing_cell_percentage']}% of total cells</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Duplicate Rows</div>
                <div class="metric-value">{kpis['duplicate_rows_count']:,}</div>
                <div class="metric-subtitle">{kpis['duplicate_percentage']}% duplicate rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Memory Footprint</div>
                <div class="metric-value">{kpis['memory_usage_formatted']}</div>
                <div class="metric-subtitle">Deep memory usage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PAGE 1: Home and Dataset Overview
# =========================================================
if selected_page == "1. Home & Dataset Overview":
    st.title("📊 Home & Dataset Overview")
    st.markdown(
        "Welcome to the **Pandas CSV Reader & Basic Data Analysis** platform. "
        "Inspect high-level metrics, investigate column profiles, view data samples, and examine detailed dataset schemas."
    )

    if st.session_state.df is None:
        st.info("Please load or upload a dataset to begin exploration.")
    else:
        render_kpi_metrics(st.session_state.df)

        st.markdown("### 📋 Interactive Column Profile Table")
        profile_df = get_column_profile(st.session_state.df)
        st.dataframe(profile_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔍 Dataset Slices & Inspection")
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔼 First Rows (`head()`)",
            "🔽 Last Rows (`tail()`)",
            "🎲 Random Sample (`sample()`)",
            "ℹ️ Schema & Types (`info()`)",
        ])

        with tab1:
            n_head = st.slider("Number of top rows to view", 5, 50, 10, key="head_slider")
            st.dataframe(get_head(st.session_state.df, n_head), use_container_width=True)
            with st.expander("View Pandas Code"):
                st.code(f"df.head({n_head})", language="python")

        with tab2:
            n_tail = st.slider("Number of bottom rows to view", 5, 50, 10, key="tail_slider")
            st.dataframe(get_tail(st.session_state.df, n_tail), use_container_width=True)
            with st.expander("View Pandas Code"):
                st.code(f"df.tail({n_tail})", language="python")

        with tab3:
            col_s1, col_s2 = st.columns([1, 1])
            with col_s1:
                n_sample = st.slider("Sample size", 1, min(100, len(st.session_state.df)), 5, key="sample_slider")
            with col_s2:
                seed = st.number_input("Random Seed", 1, 9999, 42)
            st.dataframe(get_sample(st.session_state.df, n_sample, random_state=int(seed)), use_container_width=True)
            with st.expander("View Pandas Code"):
                st.code(f"df.sample(n={n_sample}, random_state={seed})", language="python")

        with tab4:
            st.markdown("#### Schema Breakdown")
            col_types = pd.DataFrame({
                "Column": st.session_state.df.columns,
                "Pandas Dtype": [str(t) for t in st.session_state.df.dtypes],
                "Non-Null Count": [st.session_state.df[c].notna().sum() for c in st.session_state.df.columns],
                "Null Count": [st.session_state.df[c].isna().sum() for c in st.session_state.df.columns],
            })
            st.dataframe(col_types, use_container_width=True)
            with st.expander("View Pandas Code"):
                st.code("df.info()\ndf.dtypes", language="python")


# =========================================================
# PAGE 2: CSV Upload and Data Preview
# =========================================================
elif selected_page == "2. CSV Upload & Data Preview":
    st.title("📂 CSV Upload & Ingestion Settings")
    st.markdown(
        "Upload any custom CSV dataset or load synthetic benchmarking data. "
        "Supports configurable delimiters, encodings, header inference, and column pruning."
    )

    c1, c2 = st.columns([1.5, 1])

    with c1:
        st.subheader("Upload Your CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file (Max 100MB)",
            type=["csv", "txt"],
            help="Supports standard comma, tab, semicolon, pipe delimited files.",
        )

        st.markdown("#### Ingestion Configuration")
        with st.expander("⚙️ Advanced Parsing Options", expanded=False):
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                delim_choice = st.selectbox(
                    "Delimiter",
                    ["auto", ",", ";", "\\t", "|"],
                    index=0,
                    help="Select 'auto' for automatic delimiter inference",
                )
                encoding_choice = st.selectbox(
                    "File Encoding",
                    ["utf-8", "latin1", "cp1252", "iso-8859-1", "utf-16"],
                    index=0,
                )
            with col_opt2:
                has_header = st.checkbox("File has column headers", value=True)
                max_load_rows = st.number_input(
                    "Max Rows to Load (0 for unlimited)", min_value=0, max_value=500000, value=0, step=500
                )

        if uploaded_file is not None:
            if st.button("🚀 Load Uploaded Dataset", type="primary", use_container_width=True):
                nrows_val = int(max_load_rows) if max_load_rows > 0 else None
                delim_val = "\t" if delim_choice == "\\t" else delim_choice
                df_loaded, err_msg, meta = load_csv(
                    file_source=uploaded_file,
                    delimiter=delim_val,
                    encoding=encoding_choice,
                    has_header=has_header,
                    nrows=nrows_val,
                )
                if df_loaded is not None:
                    st.session_state.df = df_loaded.copy()
                    st.session_state.raw_df = df_loaded.copy()
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.cleaning_history = []
                    st.success(
                        f"Successfully loaded '{uploaded_file.name}' with {len(df_loaded):,} rows and {len(df_loaded.columns)} columns!"
                    )
                    st.rerun()
                else:
                    st.error(f"Ingestion Error: {err_msg}")

    with c2:
        st.subheader("Use Pre-Built Synthetic Datasets")
        st.info("Demonstrate features immediately with pre-configured synthetic datasets designed for analysis.")

        samples = get_sample_datasets()
        sample_choice = st.radio("Select Sample Dataset", list(samples.keys()))

        if sample_choice == "Sample Sales Dataset (Synthetic)":
            st.markdown(
                """
                - **Features:** Order ID, Date, Product, Category, Quantity, Unit Price, Region, Revenue, Rating.
                - **Use Case:** Sales performance, regional analytics, revenue aggregations.
                - **Note:** Contains intentional missing values and duplicate records for cleaning practice.
                """
            )
        else:
            st.markdown(
                """
                - **Features:** Student ID, Student Name, Subject, Marks, Attendance, Study Hours, Class Section, Grade.
                - **Use Case:** Academic distribution, study hour vs. mark correlation, grade breakdowns.
                - **Note:** Contains synthetic records with realistic variance and sample anomalies.
                """
            )

        if st.button("📥 Load Selected Sample Dataset", use_container_width=True):
            s_path = samples[sample_choice]
            df_s, err_s, meta_s = load_csv(s_path)
            if df_s is not None:
                st.session_state.df = df_s.copy()
                st.session_state.raw_df = df_s.copy()
                st.session_state.dataset_name = sample_choice
                st.session_state.cleaning_history = []
                st.success(f"Loaded '{sample_choice}' successfully!")
                st.rerun()
            else:
                st.error(f"Failed to load sample: {err_s}")

    st.markdown("---")
    st.subheader("📊 Ingested Data Preview")
    if st.session_state.df is not None:
        p_rows = st.select_slider("Preview Rows", options=[5, 10, 20, 50], value=10)
        st.dataframe(get_preview_data(st.session_state.df, p_rows), use_container_width=True)

        col_count, row_count = len(st.session_state.df.columns), len(st.session_state.df)
        st.caption(f"Currently active dataset: **{st.session_state.dataset_name}** ({row_count:,} rows, {col_count} columns)")
    else:
        st.warning("No dataset currently active.")


# =========================================================
# PAGE 3: Data Quality and Cleaning
# =========================================================
elif selected_page == "3. Data Quality & Cleaning":
    st.title("🧹 Data Quality & Interactive Cleaning")
    st.markdown(
        "Detect and remediate missing values, remove duplicate records, cast column types safely, "
        "and inspect negative values or IQR outliers. The original dataset is preserved for instant resetting."
    )

    if st.session_state.df is None:
        st.warning("Please load a dataset first.")
    else:
        render_kpi_metrics(st.session_state.df)

        clean_tab1, clean_tab2, clean_tab3, clean_tab4 = st.tabs([
            "1️⃣ Missing Values",
            "2️⃣ Duplicate Records",
            "3️⃣ Data Types Casting",
            "4️⃣ Outlier & Anomaly Validation",
        ])

        # TAB 1: Missing Values
        with clean_tab1:
            st.subheader("Missing Value Detection & Remediation")
            missing_df = detect_missing_values(st.session_state.df)

            if missing_df.empty:
                st.success("🎉 No missing values detected in the current dataset!")
            else:
                c_m1, c_m2 = st.columns([1, 1.2])
                with c_m1:
                    st.dataframe(missing_df, use_container_width=True)
                with c_m2:
                    st.plotly_chart(plot_missing_values_bar(st.session_state.df), use_container_width=True)

                st.markdown("#### Remediation Operations")
                action = st.radio(
                    "Choose Action",
                    [
                        "Fill Missing Values (Imputation)",
                        "Drop Rows with Missing Values",
                        "Drop Entire Columns with Missing Values",
                    ],
                    horizontal=True,
                )

                if action == "Fill Missing Values (Imputation)":
                    col_to_fill = st.multiselect(
                        "Select column(s) to impute",
                        missing_df["Column"].tolist(),
                        default=missing_df["Column"].tolist()[:1],
                    )
                    strategy = st.selectbox(
                        "Imputation Strategy",
                        ["mean (numeric)", "median (numeric)", "mode (most frequent)", "constant", "forward_fill", "backward_fill"],
                    )
                    custom_val = ""
                    if "constant" in strategy:
                        custom_val = st.text_input("Custom Fill Value", "Unknown")

                    if st.button("Apply Imputation", type="primary"):
                        strat_clean = strategy.split()[0]
                        before_df = st.session_state.df.copy()
                        cleaned, log = fill_missing_values(
                            st.session_state.df,
                            columns=col_to_fill,
                            strategy=strat_clean,
                            custom_value=custom_val,
                        )
                        st.session_state.df = cleaned
                        comp = get_cleaning_comparison(
                            before_df, cleaned, f"Imputed {col_to_fill} with {strategy}"
                        )
                        st.session_state.cleaning_history.append(comp)
                        st.success(f"Imputed missing values for: {col_to_fill}")
                        st.rerun()

                elif action == "Drop Rows with Missing Values":
                    drop_how = st.selectbox("Drop Condition", ["any (drop row if any value missing)", "all (drop row if all values missing)"])
                    col_subset = st.multiselect("Specific columns to check (leave blank for all columns)", st.session_state.df.columns.tolist())

                    if st.button("Drop Rows with Nulls", type="primary"):
                        before_df = st.session_state.df.copy()
                        cleaned, dropped_count = drop_missing_values(
                            st.session_state.df,
                            how="any" if "any" in drop_how else "all",
                            subset=col_subset if col_subset else None,
                        )
                        st.session_state.df = cleaned
                        comp = get_cleaning_comparison(before_df, cleaned, f"Dropped {dropped_count} rows with missing values")
                        st.session_state.cleaning_history.append(comp)
                        st.success(f"Dropped {dropped_count:,} rows containing missing values.")
                        st.rerun()

                elif action == "Drop Entire Columns with Missing Values":
                    thresh = st.slider("Missing Percentage Threshold to Drop Column", 10.0, 100.0, 50.0, step=5.0)
                    if st.button("Drop Columns Exceeding Threshold", type="primary"):
                        before_df = st.session_state.df.copy()
                        cleaned, dropped_cols = drop_missing_columns(st.session_state.df, threshold_pct=thresh)
                        if dropped_cols:
                            st.session_state.df = cleaned
                            comp = get_cleaning_comparison(before_df, cleaned, f"Dropped columns: {dropped_cols}")
                            st.session_state.cleaning_history.append(comp)
                            st.success(f"Dropped columns: {', '.join(dropped_cols)}")
                            st.rerun()
                        else:
                            st.info("No columns exceeded the specified missing threshold.")

        # TAB 2: Duplicate Records
        with clean_tab2:
            st.subheader("Duplicate Detection & Deduplication")
            dup_subset = st.multiselect(
                "Columns to evaluate for duplicates (leave blank for all columns)",
                st.session_state.df.columns.tolist(),
                key="dup_subset_select",
            )
            dup_count, dup_rows = detect_duplicates(st.session_state.df, subset=dup_subset if dup_subset else None)

            if dup_count == 0:
                st.success("🎉 No duplicate rows found based on selected criteria!")
            else:
                st.warning(f"⚠️ Found {dup_count:,} duplicate records in the dataset.")
                st.markdown("#### Duplicate Rows Preview")
                st.dataframe(dup_rows.head(20), use_container_width=True)

                col_k1, col_k2 = st.columns([1, 1])
                with col_k1:
                    keep_rule = st.selectbox("Keep Occurrence", ["first", "last"])
                with col_k2:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Remove Duplicate Rows", type="primary", use_container_width=True):
                        before_df = st.session_state.df.copy()
                        cleaned, removed = remove_duplicates(
                            st.session_state.df,
                            subset=dup_subset if dup_subset else None,
                            keep=keep_rule,
                        )
                        st.session_state.df = cleaned
                        comp = get_cleaning_comparison(before_df, cleaned, f"Removed {removed} duplicate rows (keep={keep_rule})")
                        st.session_state.cleaning_history.append(comp)
                        st.success(f"Removed {removed:,} duplicate records!")
                        st.rerun()

        # TAB 3: Data Types Casting
        with clean_tab3:
            st.subheader("Data Type Conversions")
            st.markdown("Safely transform columns to appropriate types without crashing on bad values.")

            col_c1, col_c2, col_c3 = st.columns([1.2, 1, 1])
            with col_c1:
                col_to_cast = st.selectbox("Select Column to Cast", st.session_state.df.columns.tolist())
                current_type = str(st.session_state.df[col_to_cast].dtype)
                st.caption(f"Current Type: `{current_type}`")
            with col_c2:
                target_type = st.selectbox("Target Data Type", ["integer", "float", "string", "boolean", "datetime"])
            with col_c3:
                date_fmt = None
                if target_type == "datetime":
                    date_fmt = st.text_input("Date Format (optional)", "%Y-%m-%d")
                st.write("")
                if st.button("Convert Data Type", type="primary", use_container_width=True):
                    before_df = st.session_state.df.copy()
                    cleaned, err, coerced_nulls = convert_column_type(
                        st.session_state.df,
                        column=col_to_cast,
                        target_type=target_type,
                        datetime_format=date_fmt if date_fmt else None,
                    )
                    if err:
                        st.error(f"Conversion Error: {err}")
                    else:
                        st.session_state.df = cleaned
                        detail_msg = f"Casted '{col_to_cast}' to {target_type}"
                        if coerced_nulls > 0:
                            detail_msg += f" ({coerced_nulls} invalid values converted to NaN)"
                        comp = get_cleaning_comparison(before_df, cleaned, detail_msg)
                        st.session_state.cleaning_history.append(comp)
                        st.success(f"Successfully converted '{col_to_cast}' to {target_type}!")
                        st.rerun()

        # TAB 4: Outliers & Anomalies
        with clean_tab4:
            st.subheader("Statistical Outlier & Anomaly Detection")
            st.markdown(
                "Detect values violating boundary rules or exceeding IQR thresholds. "
                "**Values are never removed automatically without explicit user authorization.**"
            )

            # Negative values check
            st.markdown("#### 1. Negative Values Check")
            neg_df = detect_negative_values(st.session_state.df)
            if neg_df.empty:
                st.success("No negative values found in numeric columns.")
            else:
                st.dataframe(neg_df, use_container_width=True)

            # IQR Outliers check
            st.markdown("#### 2. IQR Outlier Detection")
            num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                iqr_factor = st.slider("IQR Multiplier Factor (Standard is 1.5)", 1.0, 3.0, 1.5, step=0.1)
                outliers_df = detect_outliers_iqr(st.session_state.df, columns=num_cols, factor=iqr_factor)

                if outliers_df.empty:
                    st.success("No statistical outliers detected with current factor.")
                else:
                    st.dataframe(outliers_df, use_container_width=True)

                    st.markdown("#### User-Guided Outlier Remediation")
                    c_out1, c_out2, c_out3 = st.columns([1, 1, 1])
                    with c_out1:
                        target_out_col = st.selectbox("Select Column to Remediate", outliers_df["Column"].tolist())
                    with c_out2:
                        out_action = st.selectbox("Remediation Action", ["clip (Cap values to boundary limits)", "drop (Remove outlier rows)"])
                    with c_out3:
                        st.write("")
                        st.write("")
                        if st.button("Apply Outlier Action", type="primary", use_container_width=True):
                            act_key = "clip" if "clip" in out_action else "drop"
                            before_df = st.session_state.df.copy()
                            cleaned, affected = handle_outliers(
                                st.session_state.df,
                                column=target_out_col,
                                action=act_key,
                                factor=iqr_factor,
                            )
                            st.session_state.df = cleaned
                            comp = get_cleaning_comparison(
                                before_df, cleaned, f"Applied {act_key} to outliers in '{target_out_col}'"
                            )
                            st.session_state.cleaning_history.append(comp)
                            st.success(f"Outlier action '{act_key}' applied to {affected} records in '{target_out_col}'.")
                            st.rerun()
            else:
                st.info("No numeric columns available for outlier testing.")

        # Cleaning Audit Trail & Export
        st.markdown("---")
        st.subheader("📋 Cleaning Audit Trail & Download Cleaned Dataset")
        if st.session_state.cleaning_history:
            st.markdown("**Operations Log:**")
            audit_records = []
            for idx, item in enumerate(st.session_state.cleaning_history, 1):
                audit_records.append({
                    "Step": idx,
                    "Operation": item.get("operation"),
                    "Rows Delta": item.get("rows_difference"),
                    "Missing Delta": item.get("nulls_difference"),
                    "Duplicates Delta": item.get("duplicates_difference"),
                })
            st.dataframe(pd.DataFrame(audit_records), use_container_width=True)
        else:
            st.info("No cleaning operations applied yet. Dataset is in original state.")

        csv_clean_bytes = get_csv_download_buffer(st.session_state.df)
        st.download_button(
            label="💾 Download Cleaned Dataset (CSV)",
            data=csv_clean_bytes,
            file_name=f"cleaned_{st.session_state.dataset_name}.csv",
            mime="text/csv",
            type="primary",
        )


# =========================================================
# PAGE 4: Filtering and Sorting
# =========================================================
elif selected_page == "4. Filtering & Sorting":
    st.title("🔍 Filtering, Slicing & Sorting")
    st.markdown(
        "Interactive query builder demonstrating `between()`, `isin()`, `query()`, `loc[]`, `iloc[]`, "
        "`sort_values()`, and column restructuring."
    )

    if st.session_state.df is None:
        st.warning("Please load a dataset first.")
    else:
        df_work = st.session_state.df.copy()

        f_tab1, f_tab2, f_tab3 = st.tabs([
            "🎯 Interactive Multi-Filter",
            "🧩 Pandas loc[] & iloc[] Slicing",
            "⚡ Pandas query() String Expression",
        ])

        # TAB 1: Multi-Filter
        with f_tab1:
            st.subheader("Multi-Attribute Filter Builder")

            num_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df_work.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()

            c_f1, c_f2 = st.columns(2)
            applied_num_filters = []
            applied_cat_filters = []

            with c_f1:
                st.markdown("#### Numeric Range Filter")
                if num_cols:
                    chosen_num_col = st.selectbox("Numeric Column", num_cols, key="flt_num_col")
                    c_min = float(df_work[chosen_num_col].min())
                    c_max = float(df_work[chosen_num_col].max())
                    if c_min < c_max:
                        val_range = st.slider(
                            f"Range for {chosen_num_col}",
                            min_value=c_min,
                            max_value=c_max,
                            value=(c_min, c_max),
                            key="flt_num_range",
                        )
                        df_work = filter_numeric_range(df_work, chosen_num_col, val_range[0], val_range[1])
                        applied_num_filters.append({"column": chosen_num_col, "min": val_range[0], "max": val_range[1]})

            with c_f2:
                st.markdown("#### Categorical Value Filter")
                if cat_cols:
                    chosen_cat_col = st.selectbox("Categorical Column", cat_cols, key="flt_cat_col")
                    unique_vals = df_work[chosen_cat_col].dropna().unique().tolist()
                    chosen_vals = st.multiselect(
                        f"Include values in {chosen_cat_col}",
                        options=unique_vals,
                        default=unique_vals,
                        key="flt_cat_multiselect",
                    )
                    if chosen_vals and len(chosen_vals) < len(unique_vals):
                        df_work = filter_categorical_isin(df_work, chosen_cat_col, chosen_vals)
                        applied_cat_filters.append({"column": chosen_cat_col, "values": chosen_vals})

            st.markdown("#### Sorting & Column Selection")
            c_s1, c_s2, c_s3 = st.columns([1.5, 1, 1.5])
            with c_s1:
                sort_columns = st.multiselect("Sort by column(s)", df_work.columns.tolist())
            with c_s2:
                sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
            with c_s3:
                cols_to_keep = st.multiselect("Columns to display", df_work.columns.tolist(), default=df_work.columns.tolist())

            if sort_columns:
                df_work = sort_dataframe(df_work, sort_columns, ascending=(sort_order == "Ascending"))

            if cols_to_keep:
                df_work = select_columns(df_work, cols_to_keep)

            # Filter results metric
            total_orig = len(st.session_state.df)
            total_filtered = len(df_work)
            retained_pct = round((total_filtered / total_orig * 100), 2) if total_orig > 0 else 0

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Original Records", f"{total_orig:,}")
            m2.metric("Filtered Records Retained", f"{total_filtered:,}")
            m3.metric("Retention Rate", f"{retained_pct}%")

            st.markdown("#### Filtered Dataset View")
            st.dataframe(df_work, use_container_width=True)

            # Code Snippet
            with st.expander("💻 View Generated Reproducible Pandas Code"):
                code_snip = generate_pandas_filter_code(
                    numeric_filters=applied_num_filters,
                    cat_filters=applied_cat_filters,
                    sort_cols=sort_columns,
                    sort_ascending=(sort_order == "Ascending"),
                    selected_cols=cols_to_keep if len(cols_to_keep) < len(st.session_state.df.columns) else None,
                )
                st.code(code_snip, language="python")

            st.download_button(
                label="💾 Download Filtered Dataset (CSV)",
                data=get_csv_download_buffer(df_work),
                file_name="filtered_dataset.csv",
                mime="text/csv",
            )

        # TAB 2: loc[] & iloc[]
        with f_tab2:
            st.subheader("Pandas loc[] and iloc[] Demonstrator")
            c_sl1, c_sl2 = st.columns(2)

            with c_sl1:
                st.markdown("#### 1. Label-Based Slicing: `df.loc[]`")
                if num_cols:
                    l_col = st.selectbox("Condition Column", num_cols, key="loc_col")
                    l_op = st.selectbox("Condition Operator", [">", ">=", "<", "<=", "=="], key="loc_op")
                    l_val = st.number_input(
                        "Threshold Value",
                        value=float(st.session_state.df[l_col].median()),
                        key="loc_val",
                    )
                    l_display_cols = st.multiselect(
                        "Columns to project", st.session_state.df.columns.tolist(), default=st.session_state.df.columns.tolist()[:4], key="loc_cols"
                    )
                    loc_res = slice_loc(
                        st.session_state.df,
                        row_condition_col=l_col,
                        row_condition_op=l_op,
                        row_condition_val=l_val,
                        columns=l_display_cols,
                    )
                    st.caption(f"Result: {len(loc_res)} rows matched")
                    st.dataframe(loc_res.head(10), use_container_width=True)
                    st.code(f"df.loc[df['{l_col}'] {l_op} {l_val}, {l_display_cols}]", language="python")

            with c_sl2:
                st.markdown("#### 2. Position-Based Slicing: `df.iloc[]`")
                max_r = len(st.session_state.df)
                max_c = len(st.session_state.df.columns)
                r_range = st.slider("Row Index Range", 0, max_r, (0, min(10, max_r)), key="iloc_rows")
                c_range = st.slider("Column Index Range", 0, max_c, (0, min(5, max_c)), key="iloc_cols")

                iloc_res = slice_iloc(
                    st.session_state.df,
                    row_start=r_range[0],
                    row_end=r_range[1],
                    col_start=c_range[0],
                    col_end=c_range[1],
                )
                st.caption(f"Result: {iloc_res.shape[0]} rows × {iloc_res.shape[1]} columns")
                st.dataframe(iloc_res, use_container_width=True)
                st.code(f"df.iloc[{r_range[0]}:{r_range[1]}, {c_range[0]}:{c_range[1]}]", language="python")

        # TAB 3: query() Expression
        with f_tab3:
            st.subheader("Pandas `query()` String Expression")
            st.markdown(
                "Filter rows using concise boolean query syntax (e.g. `quantity > 5 and region == 'North'` or `marks >= 80`)."
            )
            default_query = ""
            if num_cols:
                default_query = f"{num_cols[0]} > {round(float(st.session_state.df[num_cols[0]].mean()), 1)}"

            query_input = st.text_input("Enter Query Expression", value=default_query)
            if query_input:
                q_res, q_err = apply_query_filter(st.session_state.df, query_input)
                if q_err:
                    st.error(f"Query Error: {q_err}")
                else:
                    st.success(f"Matched {len(q_res):,} rows ({round(len(q_res)/len(st.session_state.df)*100, 1)}%)")
                    st.dataframe(q_res, use_container_width=True)
                    st.code(f"df.query(\"{query_input}\")", language="python")


# =========================================================
# PAGE 5: Statistical Analysis
# =========================================================
elif selected_page == "5. Statistical Analysis":
    st.title("📈 Statistical Analysis & Distributions")
    st.markdown(
        "Examine parametric and non-parametric distributions, quartiles, percentiles, skewness, kurtosis, "
        "and categorical frequency breakdowns."
    )

    if st.session_state.df is None:
        st.warning("Please load a dataset first.")
    else:
        stat_tab1, stat_tab2 = st.tabs(["🔢 Numerical Column Statistics", "🏷️ Categorical Column Statistics"])

        with stat_tab1:
            st.subheader("Numerical Statistics & Distribution Profiles")
            num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()

            if not num_cols:
                st.warning("No numeric columns found in the dataset.")
            else:
                sel_num_cols = st.multiselect("Select Numeric Columns to Analyze", num_cols, default=num_cols)
                if sel_num_cols:
                    num_summary = get_numerical_summary(st.session_state.df, sel_num_cols)
                    st.dataframe(num_summary, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Distribution Visualizations")
                    c_v1, c_v2 = st.columns(2)
                    with c_v1:
                        dist_col = st.selectbox("Select Column for Histogram & Box Plot", sel_num_cols)
                        st.plotly_chart(
                            plot_histogram(st.session_state.df, dist_col, marginal="box"),
                            use_container_width=True,
                        )
                    with c_v2:
                        cat_cols = st.session_state.df.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()
                        group_by_cat = st.selectbox("Group Box Plot By Category (optional)", [None] + cat_cols)
                        st.plotly_chart(
                            plot_box(st.session_state.df, y_col=dist_col, x_col=group_by_cat),
                            use_container_width=True,
                        )

        with stat_tab2:
            st.subheader("Categorical Frequency Breakdowns")
            cat_cols = st.session_state.df.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()

            if not cat_cols:
                st.warning("No categorical columns found in the dataset.")
            else:
                cat_summary = get_categorical_summary(st.session_state.df, cat_cols)
                st.dataframe(cat_summary, use_container_width=True)

                st.markdown("---")
                c_c1, c_c2 = st.columns([1, 1.3])
                with c_c1:
                    chosen_cat = st.selectbox("Explore Value Counts for Column", cat_cols)
                    top_n = st.slider("Top N categories to display", 5, 30, 10)
                    vc_df = get_column_value_counts(st.session_state.df, chosen_cat, top_n)
                    st.dataframe(vc_df, use_container_width=True)
                with c_c2:
                    st.plotly_chart(plot_bar_chart(st.session_state.df, x_col=chosen_cat), use_container_width=True)


# =========================================================
# PAGE 6: Grouping and Aggregation
# =========================================================
elif selected_page == "6. Grouping & Aggregation":
    st.title("🗂️ Grouping, Aggregation & Pivot Tables")
    st.markdown(
        "Aggregate data across dimensions using `groupby()`, `agg()`, `pivot_table()`, and `crosstab()`. "
        "Dynamic suggestions are calculated automatically from your schema."
    )

    if st.session_state.df is None:
        st.warning("Please load a dataset first.")
    else:
        agg_tab1, agg_tab2, agg_tab3 = st.tabs([
            "📊 Multi-Column GroupBy Aggregator",
            "🔄 Interactive Pivot Table",
            "📐 Frequency Cross-Tabulation",
        ])

        all_cols = st.session_state.df.columns.tolist()
        cat_cols = st.session_state.df.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()
        num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()

        # TAB 1: Groupby
        with agg_tab1:
            st.subheader("Dynamic GroupBy & Aggregations")

            # Smart Suggestions Banner
            suggestions = get_dynamic_aggregation_suggestions(st.session_state.df)
            if suggestions:
                st.markdown("**💡 Schema-Aware Recipes:**")
                cols_sug = st.columns(len(suggestions))
                for i, sug in enumerate(suggestions):
                    with cols_sug[i]:
                        if st.button(sug["label"], key=f"sug_btn_{i}", use_container_width=True):
                            st.session_state["agg_sel_groups"] = sug["group_cols"]
                            st.session_state["agg_sel_targets"] = sug["target_cols"]
                            st.session_state["agg_sel_funcs"] = sug["agg_funcs"]

            c_g1, c_g2, c_g3 = st.columns([1.2, 1.2, 1])
            with c_g1:
                default_grp = st.session_state.get("agg_sel_groups", [cat_cols[0]] if cat_cols else [all_cols[0]])
                sel_group_cols = st.multiselect("Group By Column(s)", all_cols, default=default_grp)
            with c_g2:
                default_tgt = st.session_state.get("agg_sel_targets", [num_cols[0]] if num_cols else [all_cols[-1]])
                sel_target_cols = st.multiselect("Target Numeric Column(s)", num_cols if num_cols else all_cols, default=default_tgt)
            with c_g3:
                default_fn = st.session_state.get("agg_sel_funcs", ["sum", "mean"])
                sel_agg_funcs = st.multiselect(
                    "Aggregation Functions",
                    ["sum", "mean", "median", "count", "min", "max", "std"],
                    default=default_fn,
                )

            if sel_group_cols and sel_target_cols and sel_agg_funcs:
                grouped_res = perform_groupby(
                    st.session_state.df,
                    group_columns=sel_group_cols,
                    target_columns=sel_target_cols,
                    agg_functions=sel_agg_funcs,
                )

                st.markdown("#### Aggregation Results Table")
                st.dataframe(grouped_res, use_container_width=True)

                c_d1, c_d2 = st.columns([1, 1])
                with c_d1:
                    with st.expander("💻 View Equivalent Pandas Code"):
                        agg_dict_repr = {t: sel_agg_funcs for t in sel_target_cols}
                        st.code(
                            f"df.groupby({sel_group_cols}).agg({agg_dict_repr}).reset_index()",
                            language="python",
                        )
                with c_d2:
                    st.download_button(
                        label="💾 Download Aggregation Results (CSV)",
                        data=get_csv_download_buffer(grouped_res),
                        file_name="aggregated_results.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Select at least one grouping column, one target column, and one aggregation function.")

        # TAB 2: Pivot Table
        with agg_tab2:
            st.subheader("Pandas `pivot_table()` Builder")
            if len(all_cols) >= 3:
                c_p1, c_p2, c_p3, c_p4 = st.columns(4)
                with c_p1:
                    p_index = st.selectbox("Row Index Column", cat_cols if cat_cols else all_cols, index=0)
                with c_p2:
                    remaining_cols = [c for c in (cat_cols if cat_cols else all_cols) if c != p_index]
                    p_col = st.selectbox("Column Headers", remaining_cols if remaining_cols else all_cols, index=0)
                with c_p3:
                    p_val = st.selectbox("Value Column", num_cols if num_cols else all_cols, index=0)
                with c_p4:
                    p_func = st.selectbox("Aggregation Function", ["mean", "sum", "count", "median", "max", "min"])

                pivot_res = create_pivot_table(
                    st.session_state.df,
                    index_col=p_index,
                    columns_col=p_col,
                    values_col=p_val,
                    aggfunc=p_func,
                )
                if not pivot_res.empty:
                    st.dataframe(pivot_res, use_container_width=True)
                    with st.expander("💻 View Equivalent Pandas Code"):
                        st.code(
                            f"pd.pivot_table(df, index='{p_index}', columns='{p_col}', values='{p_val}', aggfunc='{p_func}', fill_value=0)",
                            language="python",
                        )
                else:
                    st.warning("Could not construct pivot table with chosen combination.")
            else:
                st.info("Dataset requires at least 3 columns to build a pivot table.")

        # TAB 3: Crosstab
        with agg_tab3:
            st.subheader("Categorical Cross-Tabulation (`crosstab()`)")
            if len(cat_cols) >= 2:
                c_ct1, c_ct2, c_ct3 = st.columns(3)
                with c_ct1:
                    ct_row = st.selectbox("Row Category", cat_cols, index=0)
                with c_ct2:
                    other_cats = [c for c in cat_cols if c != ct_row]
                    ct_col = st.selectbox("Column Category", other_cats, index=0)
                with c_ct3:
                    ct_norm = st.selectbox("Normalize", [None, "all", "index", "columns"])

                ct_res = create_crosstab(st.session_state.df, ct_row, ct_col, normalize=ct_norm)
                if not ct_res.empty:
                    st.dataframe(ct_res, use_container_width=True)
                    with st.expander("💻 View Equivalent Pandas Code"):
                        st.code(
                            f"pd.crosstab(df['{ct_row}'], df['{ct_col}'], normalize={repr(ct_norm)})",
                            language="python",
                        )
            else:
                st.info("At least two categorical columns are required for cross-tabulation.")


# =========================================================
# PAGE 7: Data Visualization
# =========================================================
elif selected_page == "7. Data Visualization":
    st.title("📈 Interactive Visualizations")
    st.markdown(
        "Build dynamic, interactive Plotly visualizations with custom encodings, groupings, "
        "and correlation matrices."
    )

    if st.session_state.df is None:
        st.warning("Please load a dataset first.")
    else:
        df_vis = st.session_state.df
        all_cols = df_vis.columns.tolist()
        num_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_vis.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()

        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Bar Chart",
                "Line Chart",
                "Pie / Donut Chart",
                "Histogram",
                "Scatter Plot",
                "Box Plot",
                "Correlation Heatmap",
            ],
        )

        st.markdown("---")

        if chart_type == "Bar Chart":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                bx = st.selectbox("X-Axis (Category)", all_cols, index=0)
            with c2:
                by = st.selectbox("Y-Axis Metric (Optional)", [None] + num_cols)
            with c3:
                bcolor = st.selectbox("Color / Group By (Optional)", [None] + cat_cols)
            with c4:
                bmode = st.selectbox("Bar Mode", ["group", "stack", "relative"])

            fig = plot_bar_chart(df_vis, x_col=bx, y_col=by, color_col=bcolor, barmode=bmode)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line Chart":
            c1, c2, c3 = st.columns(3)
            with c1:
                lx = st.selectbox("X-Axis (Time / Order)", all_cols, index=0)
            with c2:
                ly = st.selectbox("Y-Axis (Numeric Metric)", num_cols, index=0 if num_cols else 0)
            with c3:
                lcolor = st.selectbox("Color Segment (Optional)", [None] + cat_cols)

            fig = plot_line_chart(df_vis, x_col=lx, y_col=ly, color_col=lcolor)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie / Donut Chart":
            c1, c2, c3 = st.columns(3)
            with c1:
                pnames = st.selectbox("Category Dimension", cat_cols if cat_cols else all_cols)
            with c2:
                pvals = st.selectbox("Values Weight (Optional)", [None] + num_cols)
            with c3:
                phole = st.slider("Hole Size (Donut effect)", 0.0, 0.7, 0.35, step=0.05)

            fig = plot_pie_chart(df_vis, names_col=pnames, values_col=pvals, hole=phole)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram":
            c1, c2, c3 = st.columns(3)
            with c1:
                hx = st.selectbox("Numeric Column", num_cols if num_cols else all_cols)
            with c2:
                hbins = st.slider("Number of Bins", 5, 100, 30)
            with c3:
                hcolor = st.selectbox("Split by Color (Optional)", [None] + cat_cols)

            fig = plot_histogram(df_vis, x_col=hx, nbins=hbins, color_col=hcolor)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot":
            if len(num_cols) >= 2:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    sx = st.selectbox("X-Axis (Numeric)", num_cols, index=0)
                with c2:
                    sy = st.selectbox("Y-Axis (Numeric)", num_cols, index=1 if len(num_cols) > 1 else 0)
                with c3:
                    scolor = st.selectbox("Color By (Optional)", [None] + cat_cols)
                with c4:
                    trend = st.selectbox("Trendline", [None, "ols"])

                fig = plot_scatter(df_vis, x_col=sx, y_col=sy, color_col=scolor, trendline=trend)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("At least two numeric columns are required for a scatter plot.")

        elif chart_type == "Box Plot":
            c1, c2, c3 = st.columns(3)
            with c1:
                by = st.selectbox("Value Column (Numeric)", num_cols if num_cols else all_cols)
            with c2:
                bx = st.selectbox("Group By Dimension (Optional)", [None] + cat_cols)
            with c3:
                bpts = st.selectbox("Points Display", ["outliers", "all", False])

            fig = plot_box(df_vis, y_col=by, x_col=bx, points=bpts)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Correlation Heatmap":
            st.markdown(
                "> **Statistical Note:** Correlation assesses linear co-movement between numeric variables. "
                "**Correlation does not imply causation.**"
            )
            if len(num_cols) >= 2:
                sel_corr_cols = st.multiselect("Select Numeric Columns for Correlation Matrix", num_cols, default=num_cols)
                fig = plot_correlation_heatmap(df_vis, numeric_cols=sel_corr_cols)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least two numeric columns to generate a correlation heatmap.")


# =========================================================
# PAGE 8: Export and Analysis Report
# =========================================================
elif selected_page == "8. Export & Analysis Report":
    st.title("📑 Export & Automated Analysis Report")
    st.markdown(
        "Generate automated, evidence-backed insights derived directly from data calculations. "
        "Download clean datasets, analytical summaries, and complete markdown reports."
    )

    if st.session_state.df is None:
        st.warning("Please load a dataset first.")
    else:
        insights = generate_automated_insights(st.session_state.df)
        insights_df = generate_insights_dataframe(insights)

        st.subheader("💡 Dynamic Automated Insights")
        st.markdown(
            "Every finding below is derived at runtime from the active dataset in memory. "
            "No hardcoded assertions."
        )

        for item in insights:
            sev = item.get("severity", "Info")
            badge_class = "badge-primary"
            if sev == "Success":
                badge_class = "badge-success"
            elif sev == "Warning":
                badge_class = "badge-warning"

            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom: 10px;">
                    <span class="badge-tag {badge_class}">{item.get('category')}</span>
                    <strong>{item.get('title')}</strong> &nbsp;|&nbsp; <code>{item.get('metric_value')}</code>
                    <p style="margin-top: 6px; margin-bottom: 0; color: #334155; font-size: 0.9rem;">
                        {item.get('finding')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("📥 Export Center")

        c_exp1, c_exp2 = st.columns(2)

        with c_exp1:
            st.markdown("#### 1. Download Current Active Dataset")
            st.caption(f"Includes all cleaning and transformations ({len(st.session_state.df):,} rows)")
            st.download_button(
                label="💾 Download Active Dataset (CSV)",
                data=get_csv_download_buffer(st.session_state.df),
                file_name=f"active_{st.session_state.dataset_name}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

            st.markdown("#### 2. Download Summary Statistics")
            num_sum = get_numerical_summary(st.session_state.df)
            if not num_sum.empty:
                st.download_button(
                    label="📊 Download Summary Statistics (CSV)",
                    data=get_csv_download_buffer(num_sum),
                    file_name="summary_statistics.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with c_exp2:
            st.markdown("#### 3. Download Generated Insights Table")
            st.caption("Tabular summary of all data findings and metrics")
            if not insights_df.empty:
                st.download_button(
                    label="📋 Download Insights Table (CSV)",
                    data=get_csv_download_buffer(insights_df),
                    file_name="automated_insights.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown("#### 4. Download Full Markdown Report")
            md_report = generate_markdown_report(
                st.session_state.df,
                insights=insights,
                cleaning_history=st.session_state.cleaning_history,
                dataset_name=st.session_state.dataset_name,
            )
            st.download_button(
                label="📄 Download Markdown Analysis Report (.md)",
                data=md_report.encode("utf-8"),
                file_name="analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.markdown("---")
        with st.expander("👁️ Preview Full Markdown Report"):
            st.markdown(md_report)
