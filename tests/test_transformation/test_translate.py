import json

import polars as pl
import pytest

from src.transformation.translate import translate_csv_to_common_model


@pytest.fixture
def sample_csv(tmp_path):
    # Create a temporary CSV file with sample data
    csv_path = tmp_path / "sample_data.csv"
    csv_path.write_text("make,model,year\nJohn Deere,1025R,2020\n")
    return str(csv_path)


@pytest.fixture
def sample_semantic_layer(tmp_path):
    # Create a temporary semantic layer JSON file with sample data
    semantic_layer_path = tmp_path / "semantic_layer.json"
    semantic_data = {
        "equipment": {
            "make": {
                "keys": [{"org": "sample_dealer", "api_name": "make"}],
                "type": "string",
            },
            "model": {
                "keys": [{"org": "sample_dealer", "api_name": "model"}],
                "type": "string",
            },
            "year": {
                "keys": [{"org": "sample_dealer", "api_name": "year"}],
                "type": "integer",
            },
        },
    }
    semantic_layer_path.write_text(json.dumps(semantic_data))
    return str(semantic_layer_path)


def test_01_translate_csv_to_common_model(sample_csv, sample_semantic_layer):
    # Test successful translation of CSV to common model
    result = translate_csv_to_common_model(
        sample_csv,
        "sample_dealer",
        sample_semantic_layer,
        "equipment",
    )
    assert not result.is_empty()
    assert result.columns == ["make", "model", "year"]
    assert result["make"].dtype == pl.Utf8
    assert result["year"].dtype == pl.Int64


def test_02_translate_csv_no_matching_columns(sample_csv, tmp_path):
    # Test when no matching columns are found in the semantic layer
    semantic_layer_path = tmp_path / "semantic_layer_no_match.json"
    semantic_data = {
        "equipment": {
            "other_column": {
                "keys": [{"org": "sample_dealer", "api_name": "nonexistent_column"}],
            },
        },
    }
    semantic_layer_path.write_text(json.dumps(semantic_data))
    with pytest.raises(
        ValueError,
        match="No columns were found for dealer 'sample_dealer'",
    ):
        translate_csv_to_common_model(
            sample_csv,
            "sample_dealer",
            str(semantic_layer_path),
            "equipment",
        )


def test_03_translate_csv_column_types(sample_csv, sample_semantic_layer):
    # Test that columns are cast to the correct types as specified in the semantic layer
    result = translate_csv_to_common_model(
        sample_csv,
        "sample_dealer",
        sample_semantic_layer,
        "equipment",
    )
    assert result["make"].dtype == pl.Utf8
    assert result["model"].dtype == pl.Utf8
    assert result["year"].dtype == pl.Int64


def test_04_translate_csv_incorrect_path():
    # Test handling of an incorrect CSV path
    with pytest.raises(FileNotFoundError):
        translate_csv_to_common_model(
            "invalid_path.csv",
            "sample_dealer",
            "semantic_layer.json",
            "equipment",
        )


def test_05_translate_csv_empty_semantic_layer(sample_csv, tmp_path):
    # Test handling when semantic layer JSON is empty or does not contain the required fields
    semantic_layer_path = tmp_path / "empty_semantic_layer.json"
    semantic_layer_path.write_text("{}")
    with pytest.raises(KeyError):
        translate_csv_to_common_model(
            sample_csv,
            "sample_dealer",
            str(semantic_layer_path),
            "equipment",
        )


def test_06_translate_csv_invalid_data_type(sample_csv, tmp_path):
    # Test when data type conversion fails
    semantic_layer_path = tmp_path / "semantic_layer_invalid_type.json"
    semantic_data = {
        "equipment": {
            "make": {
                "keys": [{"org": "sample_dealer", "api_name": "make"}],
                "type": "boolean",  # Intentional type mismatch
            },
        },
    }
    semantic_layer_path.write_text(json.dumps(semantic_data))
    with pytest.raises(pl.InvalidOperationError):
        translate_csv_to_common_model(
            sample_csv,
            "sample_dealer",
            str(semantic_layer_path),
            "equipment",
        )


def test_07_translate_csv_column_types_datetime(tmp_path):
    # Create a temporary CSV file with sample data including datetime and date columns
    csv_path = tmp_path / "sample_data_datetime.csv"
    csv_path.write_text(
        "make,model,manufacture_date,last_service\nJohn Deere,1025R,2020-01-01,2021-06-15 14:30:00\n"
    )

    # Create a temporary semantic layer JSON file with sample data including datetime and date types
    semantic_layer_path = tmp_path / "semantic_layer_datetime.json"
    semantic_data = {
        "equipment": {
            "make": {
                "keys": [{"org": "sample_dealer", "api_name": "make"}],
                "type": "string",
            },
            "model": {
                "keys": [{"org": "sample_dealer", "api_name": "model"}],
                "type": "string",
            },
            "manufacture_date": {
                "keys": [{"org": "sample_dealer", "api_name": "manufacture_date"}],
                "type": "date",
            },
            "last_service": {
                "keys": [{"org": "sample_dealer", "api_name": "last_service"}],
                "type": "datetime",
            },
        },
    }
    semantic_layer_path.write_text(json.dumps(semantic_data))

    # Test that columns are cast to the correct types as specified in the semantic layer
    result = translate_csv_to_common_model(
        str(csv_path),
        "sample_dealer",
        str(semantic_layer_path),
        "equipment",
    )

    assert result["make"].dtype == pl.Utf8
    assert result["model"].dtype == pl.Utf8
    assert result["manufacture_date"].dtype == pl.Date
    assert result["last_service"].dtype == pl.Datetime
