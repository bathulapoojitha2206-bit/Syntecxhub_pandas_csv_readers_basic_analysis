"""
CSV Reader and Dataset Loader Module.
Handles robust CSV ingestion, encoding detection, delimiter inference,
header management, error handling, and sample dataset provisioning.
"""

from io import BytesIO, StringIO
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


def detect_delimiter(sample_text: str) -> str:
    """
    Infers the most likely delimiter from a text sample.
    Checks comma, semicolon, tab, and pipe.
    """
    delimiters = [",", ";", "\t", "|"]
    counts = {d: sample_text.count(d) for d in delimiters}
    # Return delimiter with highest frequency if > 0, default to comma
    best_delim = max(counts, key=counts.get)
    return best_delim if counts[best_delim] > 0 else ","


def load_csv(
    file_source: Union[str, Path, BytesIO, StringIO, bytes],
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
    has_header: bool = True,
    selected_columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Dict[str, Any]]:
    """
    Robustly loads a CSV file with support for encodings, delimiters,
    header toggling, and comprehensive error handling.

    Returns:
        (df, error_message, metadata_dict)
    """
    metadata: Dict[str, Any] = {
        "encoding_used": encoding,
        "delimiter_used": delimiter or ",",
        "has_header": has_header,
        "duplicate_columns_fixed": False,
        "original_columns": [],
    }

    if file_source is None:
        return None, "No file source provided.", metadata

    # Read bytes if passed as raw bytes or path
    raw_bytes: bytes
    if isinstance(file_source, bytes):
        raw_bytes = file_source
    elif isinstance(file_source, (str, Path)):
        if not os.path.exists(file_source):
            return None, f"File not found: {file_source}", metadata
        try:
            with open(file_source, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            return None, f"Failed to read file from disk: {str(e)}", metadata
    elif hasattr(file_source, "getvalue"):
        raw_bytes = file_source.getvalue()
    elif hasattr(file_source, "read"):
        current_pos = file_source.tell() if hasattr(file_source, "tell") else 0
        raw_bytes = file_source.read()
        if hasattr(file_source, "seek"):
            file_source.seek(current_pos)
    else:
        return None, "Unsupported file source type.", metadata

    if len(raw_bytes.strip()) == 0:
        return None, "The uploaded CSV file is completely empty.", metadata

    # Encodings to attempt if the specified encoding fails
    encodings_to_try = [encoding, "utf-8", "latin1", "cp1252", "iso-8859-1", "utf-16"]
    # De-duplicate while preserving order
    seen = set()
    encodings_to_try = [e for e in encodings_to_try if not (e in seen or seen.add(e))]

    decoded_text: Optional[str] = None
    used_encoding: Optional[str] = None

    for enc in encodings_to_try:
        try:
            decoded_text = raw_bytes.decode(enc)
            used_encoding = enc
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if decoded_text is None:
        return None, "Failed to decode CSV with supported encodings (UTF-8, Latin1, CP1252, UTF-16).", metadata

    metadata["encoding_used"] = used_encoding

    # Delimiter inference if not supplied or set to 'auto'
    if not delimiter or delimiter.lower() == "auto":
        first_lines = "\n".join(decoded_text.splitlines()[:10])
        effective_delim = detect_delimiter(first_lines)
    else:
        effective_delim = delimiter

    metadata["delimiter_used"] = effective_delim

    header_arg = 0 if has_header else None

    try:
        df = pd.read_csv(
            StringIO(decoded_text),
            sep=effective_delim,
            header=header_arg,
            nrows=nrows,
            skiprows=skiprows,
            on_bad_lines="skip",
        )
    except pd.errors.EmptyDataError:
        return None, "File contains no data or could not be parsed as a tabular CSV.", metadata
    except pd.errors.ParserError as pe:
        return None, f"CSV parsing error: {str(pe)}", metadata
    except Exception as e:
        return None, f"Error reading CSV: {str(e)}", metadata

    if df.empty:
        return None, "Dataset loaded but contains 0 rows of data.", metadata

    # If file has no headers, assign clean column names
    if not has_header:
        df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
        metadata["original_columns"] = list(df.columns)
    else:
        # Check raw header tokens from the first non-empty line
        first_line = decoded_text.strip().splitlines()[0] if decoded_text.strip() else ""
        raw_header_tokens = [tok.strip().strip('"').strip("'") for tok in first_line.split(effective_delim)]
        
        orig_cols = raw_header_tokens if len(raw_header_tokens) == df.shape[1] else list(df.columns)
        metadata["original_columns"] = [str(c) for c in orig_cols]

        clean_cols = []
        col_counts: Dict[str, int] = {}
        has_duplicates = False

        for col in orig_cols:
            col_name = str(col).strip()
            if not col_name:
                col_name = "unnamed_column"

            if col_name in col_counts:
                col_counts[col_name] += 1
                clean_name = f"{col_name}_{col_counts[col_name]}"
                has_duplicates = True
            else:
                col_counts[col_name] = 0
                clean_name = col_name
            clean_cols.append(clean_name)

        df.columns = clean_cols
        metadata["duplicate_columns_fixed"] = has_duplicates

    # Column subset selection if requested
    if selected_columns:
        valid_cols = [c for c in selected_columns if c in df.columns]
        if valid_cols:
            df = df[valid_cols]
        else:
            return None, f"None of the requested columns {selected_columns} exist in the file.", metadata

    return df, None, metadata


def get_sample_datasets() -> Dict[str, str]:
    """
    Returns available sample datasets and their absolute paths.
    """
    base_dir = Path(__file__).resolve().parent.parent / "data"
    samples = {
        "Sample Sales Dataset (Synthetic)": str(base_dir / "sample_sales.csv"),
        "Sample Students Dataset (Synthetic)": str(base_dir / "sample_students.csv"),
    }
    return samples


def get_preview_data(
    df: pd.DataFrame, rows_count: int = 5
) -> pd.DataFrame:
    """
    Returns the first N rows for preview.
    """
    return df.head(rows_count)


def get_csv_download_buffer(df: pd.DataFrame) -> bytes:
    """
    Encodes a DataFrame into a CSV byte stream for downloading.
    """
    return df.to_csv(index=False).encode("utf-8")
