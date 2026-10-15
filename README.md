# 📊 Pandas CSV Reader & Basic Data Analysis

> An interactive, portfolio-grade data analysis platform built with **Python**, **Pandas**, **NumPy**, **Streamlit**, and **Plotly** — designed for Python / Data Analyst freshers.

---

## 📌 Project Overview

This project is a fully functional, interactive web application that demonstrates practical Pandas and data analysis skills. Users can upload any CSV dataset, inspect it, clean it, filter it, aggregate it, visualize it, and export results — all through a polished Streamlit dashboard with zero code required.

Every metric, chart, and insight is **calculated dynamically** from the uploaded dataset. Nothing is hardcoded.

---

## 🔍 Problem Statement

Data analysts spend a significant portion of their time on repetitive tasks: understanding a new dataset's shape and quality, identifying missing values, removing duplicates, slicing subsets, computing groupby summaries, and building visualizations. These steps are error-prone when done ad hoc.

This tool standardises and automates those workflows, allowing analysts to focus on interpretation rather than mechanics.

---

## 🎯 Objectives

- Demonstrate practical Pandas data wrangling skills
- Provide an interactive, beginner-friendly exploration tool
- Automate common cleaning and analysis workflows
- Generate reproducible Pandas code snippets for learning
- Create a polished portfolio project for a Data Analyst internship

---

## 🛠️ Technology Stack

| Layer | Library | Version |
|---|---|---|
| Data Manipulation | Pandas | ≥ 2.2 |
| Numerical Operations | NumPy | ≥ 1.26 |
| Web Application | Streamlit | ≥ 1.35 |
| Interactive Charts | Plotly | ≥ 5.20 |
| Testing | Pytest | ≥ 8.0 |
| Runtime | Python | 3.12 |

---

## ✨ Main Features

### 1. CSV Upload & Ingestion
- Drag-and-drop CSV upload with automatic delimiter inference (`,`, `;`, `\t`, `|`)
- Multi-encoding support (UTF-8, Latin1, CP1252, ISO-8859-1, UTF-16)
- Duplicate column name detection and renaming
- No-header mode support
- Configurable row limit (prevents overloading on large files)
- Two pre-built synthetic sample datasets for immediate exploration

### 2. Dataset Overview
- Total rows, columns, numeric/categorical/datetime counts
- Missing cell count and percentage, duplicate row count, memory footprint
- Interactive column profile table (dtype, non-null, null %, unique count, sample values)
- `head()`, `tail()`, `sample()`, and schema inspection views

### 3. Data Quality & Cleaning
- Missing value detection with visual bar chart
- Imputation strategies: mean, median, mode, constant, forward-fill, backward-fill
- Drop rows or entire columns exceeding a missing % threshold
- Duplicate detection and deduplication with `keep=first/last` control
- Safe type casting to integer, float, string, boolean, datetime
- Negative value detection across numeric columns
- IQR-based outlier detection with user-guided clip or drop remediation
- Full cleaning audit trail and before/after comparison

### 4. Filtering & Sorting
- Numeric range filters using `between()`
- Categorical multi-select using `isin()`
- Date range filtering
- Full-text search with `str.contains()`
- Multi-attribute simultaneous filtering
- Column selection and sorting
- `loc[]` and `iloc[]` slicing demonstrators
- Safe `query()` string evaluator with injection protection
- Auto-generated reproducible Pandas code snippet

### 5. Statistical Analysis
- Count, Mean, Median, Mode, Std, Variance, Min, Max, Range
- Quartiles (Q1/Q3), IQR, 5th and 95th percentiles, Skewness, Kurtosis, Sum
- Categorical value counts with frequency %
- Histograms with marginal box plots, grouped box plots

### 6. Grouping & Aggregation
- Multi-column `groupby()` with multiple aggregation functions simultaneously
- `agg()` with sum, mean, median, count, min, max, std
- Dynamic schema-aware aggregation recipe suggestions
- `pivot_table()` builder with configurable index/column/value/aggfunc
- `crosstab()` with optional normalization

### 7. Data Visualization
- Bar charts (grouped/stacked)
- Line charts with time-series ordering
- Pie/Donut charts
- Histograms with marginal distributions
- Scatter plots with optional OLS trendlines
- Box plots with grouping
- Pearson Correlation Heatmap (with correlation ≠ causation disclaimer)
- Missing values bar chart

### 8. Export & Analysis Report
- Download active (cleaned/filtered) dataset as CSV
- Download summary statistics as CSV
- Download automated insights table as CSV
- Download comprehensive Markdown analysis report
- Per-page individual download buttons throughout

---

## 📁 Project Structure

```
pandas-csv-analysis/
├── app.py                          # Main Streamlit dashboard (8 pages)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git exclusions
├── data/
│   ├── sample_sales.csv            # Synthetic retail sales dataset (100 rows)
│   └── sample_students.csv         # Synthetic academic records dataset (100 rows)
├── src/
│   ├── __init__.py
│   ├── csv_reader.py               # CSV ingestion, encoding, delimiter handling
│   ├── data_overview.py            # KPIs, column profiling, statistical summaries
│   ├── data_cleaning.py            # Missing values, duplicates, type casting, outliers
│   ├── filtering.py                # loc, iloc, query, isin, between, sort
│   ├── aggregation.py              # groupby, agg, pivot_table, crosstab
│   ├── visualization.py            # Plotly chart builders
│   └── report_generator.py         # Automated insights & markdown report generation
├── outputs/                        # Exported CSV files (gitignored)
└── tests/
    ├── __init__.py
    ├── test_csv_reader.py           # 11 tests for CSV loading
    ├── test_data_cleaning.py        # 13 tests for cleaning operations
    ├── test_filtering.py            # 10 tests for filtering & sorting
    ├── test_aggregation.py          # 6 tests for groupby & pivots
    └── test_report_and_overview.py  # 7 tests for overview & report generation
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11 or higher installed
- pip package manager

### Steps

```bash
# 1. Clone or download the project folder
cd "pandas-csv-analysis"

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your default browser at:
```
http://localhost:8501
```

The application auto-loads the **Sample Sales Dataset** on first launch. Use the sidebar to switch datasets or upload your own.

---

## 📄 Supported CSV Formats & Limitations

| Feature | Supported |
|---|---|
| Comma-delimited (`,`) | ✅ |
| Semicolon-delimited (`;`) | ✅ |
| Tab-delimited (`\t`) | ✅ |
| Pipe-delimited (`\|`) | ✅ |
| UTF-8 encoding | ✅ |
| Latin1 / CP1252 encoding | ✅ (automatic fallback) |
| UTF-16 | ✅ (automatic fallback) |
| Files without headers | ✅ (toggle in settings) |
| Duplicate column names | ✅ (auto-renamed) |
| Max recommended file size | 100 MB |
| Streaming / chunked loading | ❌ (not yet implemented) |
| Excel (.xlsx) files | ❌ (CSV only) |
| Nested / JSON-embedded CSV | ❌ |

---

## 💡 Example Use Cases

### Sales Analysis
1. Load `sample_sales.csv`
2. Page 3: Remove duplicates and fill missing region values with mode
3. Page 4: Filter to Electronics category with revenue > 100
4. Page 6: GroupBy region, aggregate revenue (sum, mean)
5. Page 7: Bar chart of total revenue by category
6. Page 8: Download cleaned dataset + Markdown report

### Academic Performance Analysis
1. Load `sample_students.csv`
2. Page 5: Analyse marks distribution per subject
3. Page 7: Scatter plot of study_hours vs marks with OLS trendline
4. Page 6: Pivot table — Section (row) × Subject (col) → avg marks

---

## 📸 Screenshots

> Run the application locally and navigate each of the 8 pages to explore the full UI.

| Page | Description |
|---|---|
| Home | KPI ribbon + column profile table + head/tail/sample/schema tabs |
| CSV Upload | File uploader with advanced parsing options |
| Data Cleaning | Missing value heatmap, imputation controls, audit trail |
| Filtering | Multi-filter builder, loc/iloc demonstrators, query() evaluator |
| Statistics | Numerical summary table, histograms, grouped box plots |
| Aggregation | GroupBy builder, pivot table, crosstab |
| Visualization | 7 chart types with axis/color/aggregation controls |
| Export | Insights cards, all download buttons, Markdown report preview |

---

## 🧪 Testing Instructions

```bash
# Run all 47 tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_data_cleaning.py -v

# Run with short summary
pytest tests/ -q
```

**All 47 tests pass.** Test coverage includes:
- CSV loading with various encodings, delimiters, edge cases
- Empty file, nonexistent file, no-header files, duplicate column names
- Missing value detection, imputation, row/column dropping
- Duplicate detection and deduplication
- Boolean, float, integer, datetime type conversions
- Numeric range, categorical, date range, and text filters
- loc, iloc slicing, query() evaluation, sort
- GroupBy single/multi column, pivot table, crosstab
- Dataset KPIs, column profiling, statistical summaries
- Automated insight generation and Markdown report generation

---

## 🚀 Deployment Instructions (Render)

1. Push the project to a GitHub repository.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Configure:

| Setting | Value |
|---|---|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| **Environment** | Python 3 |

4. Set `app.py` and `requirements.txt` at the repository root (already configured).
5. Click **Deploy**.

> ⚠️ Note: The `data/` directory with sample datasets must be committed to the repository for the pre-built samples to load on deployment.

---

## 📚 Important Pandas Methods Used

| Method | Purpose in this Project |
|---|---|
| `pd.read_csv()` | Core CSV ingestion with delimiter, encoding, header options |
| `df.head(n)` | Preview first N rows |
| `df.tail(n)` | Preview last N rows |
| `df.sample(n)` | Random row sample |
| `df.info()` | Schema inspection — dtypes and non-null counts |
| `df.describe()` | Statistical summary for numeric columns |
| `df.describe(include="object")` | Summary for categorical columns |
| `df.isna()` / `df.notna()` | Missing value boolean masks |
| `df.isna().sum()` | Count missing values per column |
| `df.dropna()` | Drop rows/columns with missing values |
| `df.fillna()` | Impute missing values |
| `df.drop_duplicates()` | Remove exact duplicate rows |
| `df.duplicated()` | Boolean mask for duplicate rows |
| `df.astype()` | Convert column data types |
| `pd.to_numeric()` | Parse strings to numeric with error coercion |
| `pd.to_datetime()` | Parse strings to datetime with error coercion |
| `df.loc[]` | Label-based row and column selection |
| `df.iloc[]` | Integer position-based slicing |
| `df.query()` | Boolean query string filtering |
| `df[col].isin(vals)` | Membership filter |
| `df[col].between(a, b)` | Range filter |
| `df[col].str.contains()` | Text search filter |
| `df.sort_values()` | Sort by one or more columns |
| `df.rename()` | Rename columns |
| `df.drop()` | Remove rows or columns |
| `df.groupby()` | Group rows by one or more columns |
| `df.agg()` | Apply multiple aggregation functions |
| `pd.pivot_table()` | Reshape data into cross-tabular summary |
| `pd.crosstab()` | Frequency cross-tabulation |
| `df.reset_index()` | Move index back to columns after groupby |
| `df.corr()` | Pearson correlation matrix between numeric columns |
| `df.value_counts()` | Frequency count of unique values |
| `df.nunique()` | Count distinct values per column |
| `df.memory_usage(deep=True)` | Measure DataFrame memory footprint |
| `df.to_csv()` | Export DataFrame to CSV format |

---

## 🔮 Future Improvements

- [ ] Excel (.xlsx) file upload support
- [ ] Chunked loading for files > 500MB
- [ ] AI-assisted natural language query interface
- [ ] Column-level data lineage tracking
- [ ] Automated PDF report generation
- [ ] Multi-dataset join and merge interface
- [ ] Custom aggregation formula builder
- [ ] Scheduled automated reporting

---

## 📝 Notes

- All datasets in `data/` are **synthetic and generated for demonstration purposes only**.
- The application does **not execute or evaluate arbitrary user-uploaded code or SQL**.
- Correlation heatmaps include a disclaimer that correlation does not imply causation.

---

*Built as a portfolio project for a Python / Data Analyst internship. 🎓*
