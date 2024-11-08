"""Reads in the raw data and performs exploratory data analysis and quality checks."""

import argparse
import os

import polars as pl

from src.transformation.translate import (
    translate_csv_to_common_model,
)


def main() -> None:
    """Read in raw data and perform EDA and quality checks."""
    dealership_name = parse_inputs()

    object_files = os.listdir("data/dealers/" + dealership_name)
    object_names = [path.split(".")[0] for path in object_files]
    objects = {}

    for file, name in zip(object_files, object_names):
        objects[name] = translate_csv_to_common_model(
            f"data/dealers/{dealership_name}/{file}",
            dealership_name,
            "./src/transformation/semantic_layer.json",
            name,
        )


def parse_inputs() -> str:
    """Parse kwargs from the command line."""
    parser = argparse.ArgumentParser(description="Process dealership name.")
    parser.add_argument(
        "--dealership-name",
        "-d",
        type=str,
        required=True,
        help="Name of the dealership",
    )
    args = parser.parse_args()

    actual_dealerships = ["koenig", "ave-plp", "greenway", "akrs"]

    if args.dealership_name.lower() in actual_dealerships:
        return args.dealership_name

    msg = (
        f"Dealership {args.dealership_name} is not implemented. "
        f"Valid options are {actual_dealerships}"
    )
    raise NotImplementedError(msg)


def eda_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Perform exploratory data analysis on the input DataFrame."""
    results = []

    # Rate of missing data
    missing_data = {col: df[col].null_count() / df.height for col in df.columns}

    for col in df.columns:
        if col == "_":
            continue
        col_data = {
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
            else:
                most_common_value_count = df[col].filter(df[col].is_null()).shape[0]
            col_data.update(
                {
                    "most_common_non_null_date": most_common_date,
                    "count_of_most_common_date": most_common_value_count,
                    "unique_dates": df[col].n_unique(),
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
            most_common_value_count = (
                df[col].filter(df[col] == most_common_value).shape[0]
            )

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
    main()
