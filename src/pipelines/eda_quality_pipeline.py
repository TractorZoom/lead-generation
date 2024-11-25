"""Reads in the raw data and performs exploratory data analysis and quality checks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import polars as pl
from pandas import ExcelWriter

from src.transformation.category import CleanMakeModelData
from src.transformation.translate import translate_csv_to_common_model

log = logging.getLogger(__name__)


@contextmanager
def suppress_stdout():  # noqa: ANN201
    """Suppress stdout."""
    original_stdout = sys.stderr
    with Path(os.devnull).open("w") as f:
        sys.stderr = f
    try:
        yield
    finally:
        # Restore original stdout
        sys.stderr.close()
        sys.stderr = original_stdout


def main() -> None:
    """Read in raw data and perform EDA and quality checks."""
    args = parse_inputs()

    dealership_name = args.dealership_name
    mapping_flag = args.mapping_check
    mapping_objects = [
        {
            "name": "dealer_stock_unit",
            "make_field": "dsu_make",
            "model_field": "dsu_model",
            "group_field": "dsu_group",
        },
        {
            "name": "customer_equipment",
            "make_field": "ce_make",
            "model_field": "ce_model",
            "group_field": "ce_group",
        },
    ]

    object_files = [
        file
        for file in os.listdir("data/dealers/" + dealership_name)
        if file.endswith(".csv")
    ]
    object_names = [path.split(".")[0].replace("-", "_") for path in object_files]
    objects = {}
    with Path("./src/transformation/semantic_layer.json").open("rb") as f:
        semantic_layer = json.load(f)

    for file, name in zip(object_files, object_names):
        objects[name] = translate_csv_to_common_model(
            f"data/dealers/{dealership_name}/{file}",
            dealership_name,
            "./src/transformation/semantic_layer.json",
            name.replace("-", "_"),
        )
    log.info("Finished translating CSV files to common model")
    with ExcelWriter(f"data/dealers/{dealership_name}/eda/eda_results.xlsx") as writer:
        for object_name, pl_df in objects.items():
            eda_pl_df = eda_polars(pl_df, semantic_layer, dealership_name, object_name)
            pd_df = eda_pl_df.to_pandas()
            pd_df.to_excel(writer, sheet_name=object_name, index=False)
    log.info("Finished EDA and saved results to Excel file")

    if mapping_flag == "y":
        log.info("Starting mapping quality check")
        for obj in mapping_objects:
            object_pl = objects[obj["name"]]
            clean_make_model_data = CleanMakeModelData()
            if clean_make_model_data.aggregated_data.shape[0] == 0:
                log.info("No aggregated dataset found. This may take a minute.")
                with suppress_stdout():
                    clean_make_model_data.create_aggregated_data(
                        input_data=object_pl,
                        make_col=obj["make_field"],
                        model_col=obj["model_field"],
                        group_col=obj["group_field"],
                    )
            run_mapping_quality(
                object_pl,
                clean_make_model_data,
                make_col=obj["make_field"],
                model_col=obj["model_field"],
                group_col=obj["group_field"],
            )


def run_mapping_quality(
    pl_df: pl.DataFrame,
    clean_make_model: CleanMakeModelData,
    make_col: str,
    model_col: str,
    group_col: str,
) -> None:
    """Run quality check of being able to match equipment make/model to TZ data.

    Parameters
    ----------
    pl_df : pl.DataFrame
        The DataFrame to check.
    clean_make_model : CleanMakeModelData
        The clean make model data object.
    make_col : str
        The column name for the make.
    model_col : str
        The column name for the model.
    group_col : str
        The column name for the group.

    Returns
    -------
    None

    """
    matched = 0
    idx = 1
    for row in pl_df.iter_rows(named=True):
        make = row[make_col] if row[make_col] else ""
        model = row[model_col] if row[model_col] else ""
        group = row[group_col] if row[group_col] else ""
        if make and model:
            result = clean_make_model.clean_make_model_data(
                make,
                model,
                group=group,
            )
            if result["best_fit_score"] > 0:
                matched += 1

        else:
            log.warning("Row %d has missing make or model", idx)

        if idx % 5000 == 0:
            log.info("Matched %d rows of %d", matched, idx)
            log.info("Total of %d rows processed of %d", idx, pl_df.height)

        idx += 1
    match_rate = matched / pl_df.height
    log.info("Make/model match rate: %f", match_rate)


def parse_inputs() -> argparse.Namespace:
    """Parse kwargs from the command line."""
    parser = argparse.ArgumentParser(description="Process dealership name.")
    parser.add_argument(
        "--dealership-name",
        "-d",
        type=str,
        required=True,
        choices=["koenig", "ave-plp", "greenway", "akrs"],
        help="Name of the dealership",
    )
    parser.add_argument(
        "--mapping-check",
        "-m",
        type=str,
        required=False,
        choices=["y", "n"],
        default="n",
        help="Whether to perform a full mapping check which can take some time (y/n)",
    )
    return parser.parse_args()


def get_salesforce_object_and_field(
    semantic_layer: dict,
    dealer: str,
    obj: str,
    field_name: str,
) -> tuple[str, str]:
    """Get the Salesforce object and field name from the semantic layer."""
    data = semantic_layer[obj][field_name]

    for key in data["keys"]:
        if key["org"] == dealer:
            return key["object"], key["api_name"]
    return "", ""


def eda_polars(
    df: pl.DataFrame,
    semantic_layer: dict,
    dealer: str,
    object_name: str,
) -> pl.DataFrame:
    """Perform exploratory data analysis on the input DataFrame."""
    results = []

    # Rate of missing data
    missing_data = {col: df[col].null_count() / df.height for col in df.columns}
    for col in df.columns:
        sf_object, sf_field = get_salesforce_object_and_field(
            semantic_layer,
            dealer,
            object_name,
            col,
        )
        if col == "_":
            continue
        col_data = {
            "anvil_object": sf_object,
            "anvil_field": sf_field,
            "field_name": col,
            "missing_data_rate": missing_data[col],
        }
        if (
            df[col].dtype == pl.Int32
            or df[col].dtype == pl.Float64
            or df[col].dtype == pl.Int64
        ):  # Checks for numeric types
            col_data.update(
                {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "median": df[col].median(),
                    "25th_percentile": df[col].quantile(0.25),
                    "75th_percentile": df[col].quantile(0.75),
                    "zero_count": df[col].filter(df[col] == 0).shape[0],
                    "non_null_count": df[col].filter(df[col].is_not_null()).shape[0],
                },
            )

        if df[col].dtype == pl.Date or df[col].dtype == pl.Datetime:
            most_common_date = (
                df[col].filter(~df[col].is_null()).mode().to_list()[0]
                if len(df[col].filter(~df[col].is_null()).mode()) > 0
                else None
            )
            if most_common_date:
                most_common_value_count = (
                    df[col].filter(df[col] == most_common_date).shape[0]
                )
                most_common_date = most_common_date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                most_common_value_count = df[col].filter(df[col].is_null()).shape[0]
            earliest_date = df[col].min()
            if earliest_date:
                earliest_date = earliest_date.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore

            col_data.update(
                {
                    "most_common_non_null_date": most_common_date,
                    "count_of_most_common_date": most_common_value_count,
                    "unique_dates": df[col].n_unique(),
                    "earliest_date": earliest_date,
                },
            )

        if (
            df[col].dtype == "String" or df[col].dtype == pl.Utf8
        ):  # Checks for categorical/string types
            most_common_value = (
                df[col].filter(~df[col].is_null()).mode().to_list()[0]
                if len(df[col].filter(~df[col].is_null()).mode()) > 0
                else None
            )
            if most_common_value:
                most_common_value_count = (
                    df[col].filter(df[col] == most_common_value).shape[0]
                )
            else:
                most_common_value_count = df[col].filter(df[col].is_null()).shape[0]

            col_data.update(
                {
                    "most_common_non_null_value": most_common_value,
                    "count_of_most_common_value": most_common_value_count,
                    "unique_values": df[col].n_unique(),
                },
            )

        if df[col].dtype == pl.Boolean:
            col_data.update(
                {
                    "true_count": df[col].filter(df[col]).shape[0],
                    "false_count": df[col].filter(~df[col]).shape[0],
                    "missing_count": df[col].filter(df[col].is_null()).shape[0],
                },
            )

        results.append(col_data)

    return pl.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
