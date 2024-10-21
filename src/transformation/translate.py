"""
Functions required to process raw data through semantic layer transformation
"""

import polars as pl
import json

def translate_csv_to_common_model(csv_path: str, dealer: str, semantic_layer_path: str) -> pl.DataFrame:
    """
    Translates a raw CSV for a given dealer into a common data model using the semantic layer.
    
    Parameters:
    csv_path (str): The path to the raw CSV file.
    dealer (str): The dealer name used to identify field mappings in the semantic layer.
    semantic_layer_path (str): The path to the semantic layer JSON file.
    
    Returns:
    pl.DataFrame: Translated Polars DataFrame in the common data model format.
    """
    
    # Load the raw CSV data into a Polars DataFrame
    df = pl.read_csv(csv_path)
    
    # Load the semantic layer JSON
    with open(semantic_layer_path, 'r', encoding="utf-8") as f:
        semantic_layer = json.load(f)

    # Create a dictionary to store the mapping of raw columns to common model columns
    column_mapping = {}

    # Iterate through the semantic layer to build the column mapping for the dealer
    for _, fields in semantic_layer.items():
        for field_name, field_data in fields.items():
            # Find the mapping for the specified dealer
            for key_mapping in field_data["keys"]:
                if key_mapping["org"] == dealer:
                    raw_column_name = key_mapping["api_name"]
                    # Store the mapping between the raw column and the common model column (field_name)
                    column_mapping[raw_column_name] = field_name

    # Check for columns in the raw CSV that can be translated using the semantic layer
    common_model_columns = []
    for col in df.columns:
        if col in column_mapping:
            # Rename the column to its common model name
            common_model_columns.append((col, column_mapping[col]))

    # If no columns were matched, raise an error
    if not common_model_columns:
        raise ValueError(f"No columns were found for dealer '{dealer}' in the semantic layer.")

    # Rename the columns in the DataFrame according to the semantic layer mapping
    df_translated = df.rename(dict(common_model_columns))
    
    return df_translated

def translate_koenig_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Translates the columns of a Polars DataFrame to the common data model for Koenig dealers.
    
    Parameters:
    df (pl.DataFrame): The Polars DataFrame to translate.
    
    Returns:
    """

    # apply translate_koenig_dealer_loyalty function to the column Anvil__Competitive_Owner__c
    df = df.with_column(
        df.apply(translate_koenig_dealer_loyalty, in_place=False)
    )


def translate_koenig_dealer_loyalty(row) -> str:
    """
    Translates the columns of a Polars DataFrame to the common data model for Koenig dealers.
    
    Parameters:
    df (pl.DataFrame): The Polars DataFrame to translate.
    """

    if row["Anvil__Competitive_Owner__c"] == "IC":
        return "In-Line Competitor"
    elif row["Anvil__Competitive_Owner__c"] == "IK":
        return "In-Line Koenig"
    elif row["Anvil__Competitive_Owner__c"] == "IS":
        return "In-Line Shared"
    elif row["Anvil__Competitive_Owner__c"] == "OC":
        return "Other Competitor"
    elif row["Anvil__Competitive_Owner__c"] == "RC":
        return "Rainbow Competitor"
    elif row["Anvil__Competitive_Owner__c"] == "RK":
        return "Rainbow Koenig"
    elif row["Anvil__Competitive_Owner__c"] == "CNV":
        return "CNV"
    else:
        return "No"

    