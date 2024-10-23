"""Contains functions to translate raw data into a common data model."""

import json
from pathlib import Path

import polars as pl


def translate_csv_to_common_model(
    csv_path: str, dealer: str, semantic_layer_path: str
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
    df_translate = pl.read_csv(csv_path)

    # Load the semantic layer JSON
    with Path(semantic_layer_path).open(encoding="utf-8") as f:
        semantic_layer = json.load(f)

    # Create a dictionary to store the mapping of raw columns to common model columns
    column_mapping = {}

    for fields in semantic_layer.values():
        for field_name, field_data in fields.items():
            for key_mapping in field_data["keys"]:
                if key_mapping["org"] == dealer:
                    raw_column_name = key_mapping["api_name"]
                    column_mapping[raw_column_name] = field_name

    # Check for columns in the raw CSV that can be translated using the semantic layer
    common_model_columns = [
        (col, column_mapping[col])
        for col in df_translate.columns
        if col in column_mapping
    ]

    # If no columns were matched, raise an error
    if not common_model_columns:
        error_message = (
            f"No columns were found for dealer '{dealer}' in the semantic layer."
        )
        raise ValueError(error_message)

    # Rename the columns in the DataFrame according to the semantic layer mapping
    return df_translate.rename(dict(common_model_columns))


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

    updated_df = df.with_columns(
        pl.col("dsu_status").replace(stock_unit_translate_dict)
    )

    date_cols = [
        "dsu_check_in_date",
        "dsu_date_promised",
        "dsu_order_date",
        "dsu_purchase_date",
        "dsu_basic_warranty_end_date",
        "dsu_extended_warranty_end_date",
        "dsu_sales_date",
    ]
    # update datetime columns to datetime type
    return updated_df.with_columns(
        **{col: pl.col(col).str.to_date(format="%Y-%m-%d") for col in date_cols}
    )
