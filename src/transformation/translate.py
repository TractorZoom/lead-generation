"""Contains functions to translate raw data into a common data model."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def translate_csv_to_common_model(
    csv_path: str,
    dealer: str,
    semantic_layer_path: str,
    object_type: str,
) -> pl.DataFrame:
    """Translate a raw CSV for a dealer into a common data model using semantic layer.

    Parameters
    ----------
    csv_path : str
        The path to the raw CSV file.
    dealer : str
        The dealer name used to identify field mappings in the semantic layer.
    semantic_layer_path : str
        The path to the semantic layer JSON file.
    object_type : str
        The object type to translate (e.g., equipment, customer, etc.)

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

    column_mapping, column_types = create_column_mapping(semantic_layer, dealer)

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

    # Drop columns not in semantic lyayer
    columns_to_drop = [
        col for col in df_translate.columns if col not in column_mapping.values()
    ]

    df_translate = df_translate.drop(columns_to_drop)
    return translate_columns(df_translate, column_types)


def translate_columns(df_translate: pl.DataFrame, column_types: dict) -> pl.DataFrame:
    """Translate the columns of a Polars DF to the common data model.

    Parameters
    ----------
    df_translate : pl.DataFrame
        The Polars DataFrame to translate.
    column_types : dict
        A dictionary mapping column names to their data types.

    Returns
    -------
    pl.DataFrame

    """
    for col, data_type in column_types.items():
        if data_type == "date":
            df_translate = df_translate.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8) == "")
                .then(None)
                .otherwise(pl.col(col))
                .str.to_date(format="%Y-%m-%d")
                .alias(col),
            )
        elif data_type == "datetime":
            df_translate = df_translate.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8) == "")
                .then(None)
                .otherwise(pl.col(col))
                .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.3f%z", strict=False)
                .alias(col),
            )
        elif data_type == "float":
            df_translate = df_translate.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8) == "")
                .then(None)
                .otherwise(pl.col(col))
                .cast(pl.Float64)
                .alias(col),
            )
        elif data_type == "integer":
            df_translate = df_translate.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8) == "")
                .then(None)
                .otherwise(pl.col(col))
                .cast(pl.Int64)
                .alias(col),
            )
        elif data_type == "boolean":
            df_translate = df_translate.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8) == "")
                .then(None)
                .otherwise(pl.col(col))
                .cast(pl.Boolean)
                .alias(col),
            )
        elif data_type == "string":
            df_translate = df_translate.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8) == "")
                .then(None)
                .otherwise(pl.col(col))
                .cast(pl.Utf8)
                .alias(col),
            )
        else:
            continue
    return df_translate


def create_column_mapping(semantic_layer: dict, dealer: str) -> tuple[dict, dict]:
    """Create a mapping of raw columns to common model columns using the semantic layer.

    Parameters
    ----------
    semantic_layer : dict
        The semantic layer dictionary.
    dealer : str
        The dealer name used to identify field mappings in the semantic layer.

    Returns
    -------
    dict
        A dictionary mapping raw columns to common model columns.


    """
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
    return column_mapping, column_types


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
        "R": "C - Relationship Account",
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


def translate_greenway_stock_units(df: pl.DataFrame) -> pl.DataFrame:

    variant_translate_dict = {
        "2WD": "Row-Crop Tractors",
        "4WD": "Articulated Tractors",
        "AIR": "Air Seeders",
        "AMS": "ISG",
        "AUG": "Material Handling",
        "BKL": "Pending Backhoes",
        "CEX": "Mini Excavators",
        "CHP": "Landscape Equipment",
        "CLD": "Wheel Loaders",
        "CMB": "Combine",
        "CPL": "Chisel Plow",
        "CRH": "Corn Headers",
        "CTL": "Compact Track Loaders",
        "CUT": "Compact Utility Tractors",
        "DLG": "Garden Tractors",
        "DLT": "Lawn Tractors",
        "DNR": "Screens",
        "DRL": "Drills",
        "DRS": "Scraper/Dirt Pans",
        "DSK": "Disks",
        "EFM": "Mower Decks",
        "FCS": "PRECISION AG TECHNOLOGY",
        "FCV": "Cultivator",
        "FHV": "Forage Harvesters",
        "FLF": "Forklifts",
        "FLS": "Cutters & Shredders",
        "FME": "Front Mount Mowers",
        "FMX": "Material Handling",
        "FOR": "FOR",
        "GMT": "Golf Products",
        "GMW": "Grooming Mowers",
        "GUI": "Activations",
        "HDR": "Combine Harvesting",
        "HFH": "Hay & Forage",
        "HFI": "Bale Spears",
        "HRV": "Harvesters",
        "HVU": "Hay & Forage",
        "IMT": "PRECISION AG TECHNOLOGY",
        "LAT": "Landscape Equipment",
        "LDR": "Large Loaders",
        "LEM": "Landscape Equipment",
        "MAN": "Manure Spreader",
        "MCO": "MoCo",
        "MFN": "Mulch Finisher",
        "MHI": "Buckets",
        "MOW": "Mowers (Hay)",
        "MTL": "Mulch Tiller",
        "NAP": "Anhydrous Bar",
        "OPN": "Planting & Seeding",
        "OSD": "Planting & Seeding",
        "PHD": "Posthole Digger",
        "PLF": "Platforms",
        "PLT": "Planters",
        "PTI": "Landscape Equipment",
        "RAK": "Rakes",
        "RBL": "Large Tractor Blades",
        "RBR": "Landscape Equipment",
        "RCT": "Rotary Cutters",
        "RDB": "Round Balers",
        "REB": "CUT Blades",
        "RHU": "Hay & Forage",
        "RIP": "Rippers",
        "RZT": "Residential Zero Turn",
        "SBF": "Seed Bed Finisher",
        "SEQ": "Snowblowers & Attachments",
        "SKS": "Skid Steers",
        "SPA": "Self-Propelled Applicators",
        "SPD": "Material Handling",
        "SPR": "Bale Spear",
        "SPY": "Self-Propelled Sprayers",
        "SQB": "Square Balers",
        "TED": "Tedders",
        "TEL": "Telehandlers",
        "TIL": "Tillers",
        "TIM": "Pending Blades",
        "TIR": "Tires",
        "TLB": "Backhoes",
        "TRC": "Pending Tractors",
        "TRI": "Pending Attachments",
        "TRK": "Track Tractors",
        "TRL": "Trailers",
        "TRU": "Pending 1-5 Series Tractors",
        "UTV": "Utility Vehicles",
        "WAG": "Header/Grain Carts, Tenders",
        "WAP": "Pending CCE Attachments",
        "WBP": "Mowers",
        "WDO": "Hay & Forage",
        "WDP": "Windrowers",
        "WHR": "Hay & Forage",
        "WPI": "Pending CCE Attachments",
        "ZCH": "Harvester Attachments",
        "ZCW": "Construction Attachments",
        "ZHF": "Hay & Forage Attachments",
        "ZLC": "Implement Attachments",
        "ZMA": "Agriculture Implements",
        "ZMO": "Mower Attachments",
        "ZPF": "AMS Attachments",
        "ZPS": "Seeding Attachments",
        "ZSP": "Applicator Attachments",
        "ZTA": "Telehandler Attachments",
        "ZTC": "Tractor Attachments",
        "ZTL": "Tillage Attachments",
        "ZTR": "Commercial Zero Turn",
        "ZUV": "UTV Attachments",
        "ZWG": "Cart/Wagon/Tender Attachments",
        "FH": "FH",
        "FAT": "FAT",
        "CAT": "CAT",
        "OAT": "OAT",
    }

    return df.with_columns(pl.col("dsu_variant").replace(variant_translate_dict))
