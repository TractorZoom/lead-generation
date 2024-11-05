"""Contains functions to translate raw data into a common data model."""

import json
from pathlib import Path

import polars as pl


def translate_csv_to_common_model(
    csv_path: str, dealer: str, semantic_layer_path: str, object_type: str
) -> pl.DataFrame:
    """Translate a raw CSV for a given dealer into a common data model using semantic layer.

    Parameters
    ----------
    csv_path : str
        The path to the raw CSV file.
    dealer : str
        The dealer name used to identify field mappings in the semantic layer.
    semantic_layer_path : str
        The path to the semantic layer JSON file.

    Returns
    -------
    pl.DataFrame
        Translated Polars DataFrame in the common data model format.

    """
    # Load the raw CSV data into a Polars DataFrame
    df_init = pl.read_csv(csv_path, ignore_errors=True)

    # Load the semantic layer JSON
    with Path(semantic_layer_path).open(encoding="utf-8") as f:
        semantic_layer = json.load(f)
    semantic_layer = semantic_layer[object_type]

    # Create a dictionary to store the mapping of raw columns to common model columns
    column_mapping = {}
    column_types = {}
    for field_name, field_data in semantic_layer.items():
        for key_mapping in field_data["keys"]:
            if key_mapping["org"] == dealer:
                raw_column_name = key_mapping["api_name"]
                column_mapping[raw_column_name] = field_name
                if "type" in field_data:
                    column_types[field_name] = field_data["type"]

    # Check for columns in the raw CSV that can be translated using the semantic layer

    common_model_columns = [
        (col, column_mapping[col]) for col in df_init.columns if col in column_mapping
    ]

    # If no columns were matched, raise an error
    if not common_model_columns:
        error_message = (
            f"No columns were found for dealer '{dealer}' in the semantic layer."
        )
        raise ValueError(error_message)

    # Rename the columns in the DataFrame according to the semantic layer mapping
    df_translate = df_init.rename(dict(common_model_columns))

    for col, data_type in column_types.items():
        if data_type == "date":
            df_translate = df_translate.with_columns(
                pl.col(col)
                .replace("", None)
                .str.to_date(format="%Y-%m-%d", strict=False)
            )
        elif data_type == "datetime":
            df_translate = df_translate.with_columns(
                pl.col(
                    col.replace("", None).str.to_date(
                        format="%Y-%m-%d %H:%M:%S", strict=False
                    )
                )
            )
        elif data_type == "float":
            if df_translate[col].dtype == pl.Float64:
                continue
            df_translate = df_translate.with_columns(
                pl.col(col).replace("", None).cast(pl.Float64)
            )
        elif data_type == "integer":
            if df_translate[col].dtype == pl.Int64:
                continue
            else:
                df_translate = df_translate.with_columns(
                    pl.col(col).replace("", None).cast(pl.Int64)
                )
        elif data_type == "boolean":
            df_translate = df_translate.with_columns(
                pl.col(col).replace("", None).cast(pl.Boolean)
            )
        elif data_type == "string":
            if df_translate[col].dtype != pl.Utf8:
                df_translate = df_translate.with_columns(pl.col(col).cast(pl.Utf8))

            df_translate = df_translate.with_columns(pl.col(col).replace("", None))
        else:
            continue

    return df_translate


def translate_koenig_account_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Translate the columns of a Polars DF to the common data model for Koenig.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to translate.

    Returns
    -------
    pl.DataFrame
        Translated Polars DataFrame in the common data model format.

    """
    comp_translate_dict = {
        "IC": "In-Line Competitor",
        "IK": "In-Line Koenig",
        "IS": "In-Line Shared",
        "OC": "Other Competitor",
        "RC": "Rainbow Competitor",
        "RK": "Rainbow Koenig",
        "CNV": "CNV",
    }

    business_translate_dict = {
        "S": "A - Strategic Account",
        "K": "B - Key Account",
        "R": "R - Relationship Account",
        "T": "D - Transaction Account",
        "ST": "A - Turf Strategic Account",
        "KT": "K - Turf Key Account",
        "RT": "C - Turf Relationship Account",
        "TT": "D - Turf Transaction Account",
        "I": "Investigate",
    }

    equipment_customer_translate_dict = {
        "N": "New Only",
        "O": "Both, New Primary",
        "U": "Used Only",
        "IS": "Both, Used Primary",
    }

    customer_translate_dict = {
        "Strategic Partner": "Strategic Partner",
        "AS": "Ag Service Provider",
        "C": "Contractor",
        "CG": "Cash Grain",
        "D": "Dealer",
        "DA": "Dairy",
        "G": "Governmental",
        "GB": "Grain Beef",
        "GD": "Grain Dairy",
        "GH": "Grain Hogs",
        "GL": "Grain Livestock",
        "HA": "Hay",
        "L": "Landscaper",
        "LP": "Large Property Owner",
        "NF": "No Longer Farms",
        "R": "Rental",
        "SC": "Specialty Crop",
    }

    return df.with_columns(
        pl.col("customer_loyalty").replace(comp_translate_dict),
        pl.col("customer_business_class").replace(business_translate_dict),
        pl.col("type_of_equipment").replace(equipment_customer_translate_dict),
        pl.col("customer_segment").replace(customer_translate_dict),
    )


def translate_koenig_stock_unit(df: pl.DataFrame) -> pl.DataFrame:
    """Translate the columns of a Polars DF to the common data model for Koenig.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to translate.

    Returns
    -------
    pl.DataFrame
        Translated Polars DataFrame in the common data model format.

    """
    stock_unit_translate_dict = {
        "V": "Inventory",
        "O": "On Order",
        "S": "Sold",
        "P": "Presold",
        "I": "Invoiced",
        "R": "Rental",
        "T": "Transfer",
        "D": "D",
        "X": "X",
    }

    return df.with_columns(pl.col("dsu_status").replace(stock_unit_translate_dict))


def translate_keonig_customer_equipment(df: pl.DataFrame) -> pl.DataFrame:
    """Translate the columns of a Polars DF to the common data model for Koenig.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to translate.


    Returns
    -------
    pl.DataFrame
        Translated Polars DataFrame in the common data model format.

    """
    status_translate_dict = {
        "Owned": "Owned",
        "Sold": "Sold",
        "Traded": "Traded",
        "Scrapped": "Scrapped",
    }

    return df.with_columns(pl.col("ce_status").replace(status_translate_dict))


def translate_koenig_purchase_orders(df: pl.DataFrame) -> pl.DataFrame:
    """Translate the columns of a Polars DF to the common data model for Koenig.

    NOTE: This function is a placeholder and does not perform any translation.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to translate.

    Returns
    -------
    pl.DataFrame
        Translated Polars DataFrame in the common data model format.

    """
    return df


def translate_koenig_service_requests(df: pl.DataFrame) -> pl.DataFrame:
    """Translate the columns of a Polars DF to the common data model for Koenig.

    NOTE: This function is a placeholder and does not perform any translation.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to translate.

    Returns
    -------
    pl.DataFrame
        Translated Polars DataFrame in the common data model format.

    """
    return df


def translate_koenig_store(df: pl.DataFrame) -> pl.DataFrame:
    """Translate the columns of a Polars DF to the common data model for Koenig.

    NOTE: This function is a placeholder and does not perform any translation.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to translate.

    Returns
    -------
    pl.DataFrame
        Translated Polars DataFrame in the common data model format.

    """
    return df


def translate_koenig_task(df: pl.DataFrame) -> pl.DataFrame:
    """Translate the columns of a Polars DF to the common data model for Koenig.

    NOTE: This function is a placeholder and does not perform any translation.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to translate.

    Returns
    -------
    pl.DataFrame
        Translated Polars DataFrame in the common data model format.

    """
    return df


def translate_koenig_user(df: pl.DataFrame) -> pl.DataFrame:
    """Translate the columns of a Polars DF to the common data model for Koenig.

    NOTE: This function is a placeholder and does not perform any translation.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to translate.

    Returns
    -------
    pl.DataFrame
        Translated Polars DataFrame in the common data model format.

    """
    return df
