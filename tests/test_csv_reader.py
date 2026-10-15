"""
Unit Tests for CSV Reader and Ingestion Module.
"""

from io import BytesIO
import os
import pandas as pd
import pytest

from src.csv_reader import (
    load_csv,
    detect_delimiter,
    get_sample_datasets,
    get_preview_data,
    get_csv_download_buffer,
)


def test_detect_delimiter():
    assert detect_delimiter("a,b,c\n1,2,3") == ","
    assert detect_delimiter("a;b;c\n1;2;3") == ";"
    assert detect_delimiter("a\tb\tc\n1\t2\t3") == "\t"
    assert detect_delimiter("a|b|c\n1|2|3") == "|"


def test_load_csv_valid_bytes():
    content = b"col1,col2,col3\n10,apple,1.5\n20,banana,2.5\n30,cherry,3.5\n"
    df, err, meta = load_csv(content)
    assert err is None
    assert df is not None
    assert df.shape == (3, 3)
    assert list(df.columns) == ["col1", "col2", "col3"]
    assert meta["delimiter_used"] == ","
    assert meta["encoding_used"] == "utf-8"


def test_load_csv_custom_delimiter():
    content = b"name;age;city\nAlice;30;New York\nBob;25;London\n"
    df, err, meta = load_csv(content, delimiter=";")
    assert err is None
    assert df is not None
    assert df.shape == (2, 3)
    assert list(df.columns) == ["name", "age", "city"]


def test_load_csv_auto_delimiter_semicolon():
    content = b"name;age;city\nAlice;30;New York\nBob;25;London\nCharlie;35;Paris\n"
    df, err, meta = load_csv(content, delimiter="auto")
    assert err is None
    assert df is not None
    assert df.shape == (3, 3)
    assert meta["delimiter_used"] == ";"


def test_load_csv_encoding_fallback():
    # Encode with latin1 containing accents
    text = "id,name,city\n1,Ren\xe9,Montr\xe9al\n2,Jos\xe9,S\xe3o Paulo\n"
    latin1_bytes = text.encode("latin1")
    df, err, meta = load_csv(latin1_bytes, encoding="utf-8")
    assert err is None
    assert df is not None
    assert df.shape == (2, 3)


def test_load_csv_empty_file():
    empty_bytes = b""
    df, err, meta = load_csv(empty_bytes)
    assert df is None
    assert "empty" in err.lower()


def test_load_csv_nonexistent_file():
    df, err, meta = load_csv("non_existent_file_xyz_1234.csv")
    assert df is None
    assert "not found" in err.lower()


def test_load_csv_without_header():
    content = b"100,ProductA,50.0\n101,ProductB,75.0\n"
    df, err, meta = load_csv(content, has_header=False)
    assert err is None
    assert df is not None
    assert df.shape == (2, 3)
    assert list(df.columns) == ["column_1", "column_2", "column_3"]


def test_load_csv_duplicate_column_names():
    content = b"id,val,val\n1,10,100\n2,20,200\n"
    df, err, meta = load_csv(content)
    assert err is None
    assert df is not None
    assert list(df.columns) == ["id", "val", "val_1"]
    assert meta["duplicate_columns_fixed"] is True


def test_sample_datasets_exist():
    samples = get_sample_datasets()
    assert "Sample Sales Dataset (Synthetic)" in samples
    assert "Sample Students Dataset (Synthetic)" in samples

    for name, path in samples.items():
        assert os.path.exists(path)
        df, err, meta = load_csv(path)
        assert err is None
        assert df is not None
        assert len(df) > 10


def test_preview_and_download_buffer():
    content = b"a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n11,12\n"
    df, _, _ = load_csv(content)
    preview = get_preview_data(df, rows_count=3)
    assert len(preview) == 3

    buf = get_csv_download_buffer(df)
    assert isinstance(buf, bytes)
    assert b"a,b" in buf
