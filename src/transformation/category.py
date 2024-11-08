"""Contains functionality to map Anvil equipment data to TZ Cat + Subcat."""

import logging

import polars as pl
from sentence_transformers import SentenceTransformer, util

from src.utils.io import read_from_databricks

SENTENCE_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

MAX_CHARACTERS_IN_ACROYNM = 3

MAKE_MODEL_QUERY = """SELECT m2.id
     , m1.name AS make
     , m2.model
     , cat.name AS category
     , sub.name AS subcategory
     , COUNT(l.id) AS total_units
FROM prod_mysql.silver_tractor_prod.make m1
JOIN prod_mysql.silver_tractor_prod.make_model m2 ON m1.id = m2.make_id
JOIN prod_mysql.silver_tractor_prod.lot_category cat ON cat.id = m2.category
JOIN prod_mysql.silver_tractor_prod.lot_subcategory sub ON sub.id = m2.subcategory
JOIN prod_mysql.silver_tractor_prod.lots l ON l.category_id = cat.id AND l.subcategory_id = sub.id
WHERE m2.status = 1 
GROUP BY ALL
ORDER BY total_units DESC"""

log = logging.getLogger(__name__)


class CleanMakeModelData:
    def __init__(self) -> None:
        self.make_model_data = self.get_make_model_data()
        self.aggregated_data = pl.DataFrame()

    def clean_make_model_data(
        self,
        make: str,
        model: str,
        group: str = "",
    ) -> dict:
        """Correct and enrich make model data.

        Takes in a make, model, description & return corrected
        make, model, category, and subcategory based on TZ data. This
        function is the primary entry point

        Parameters
        ----------
        make : str
            The make of the equipment.
        model : str
            The model of the equipment.
        description : str
            The description of the equipment.

        Returns
        -------
        dict
            The cleaned make, model, category, and subcategory data.

        """
        # check for exact match in make model data
        exact_match = self._check_match(
            make, model, group=group, use_semantic_check=True,
        )
        if exact_match:
            return exact_match[0]

        # check for acronym in make
        if len(make) <= MAX_CHARACTERS_IN_ACROYNM:
            synonym_make = self.make_synonym_list(make)
            match_check = self._check_match(
                synonym_make,
                model,
                group=group,
                use_semantic_check=True,
            )
            if match_check:
                return match_check[0]

        # check for most most likely based on aggregated data
        if self.aggregated_data.shape[0] > 0:
            return self.check_aggregated_data(make, model, group)

        return {}

    def create_aggregated_data(
        self,
        input_data: pl.DataFrame,
        make_col: str = "dsu_make",
        model_col: str = "dsu_model",
        group_col: str = "dsu_group",
    ) -> None:
        """Create aggregated data for make model data."""
        temp_data = []
        make_model_data = input_data[[make_col, model_col]].unique()
        # Run _check_match for each row in input_data
        for i in range(make_model_data.shape[0]):
            make = make_model_data[make_col][i]
            model = make_model_data[model_col][i]
            if not make or not model:
                continue
            match = self._check_match(make, model, use_semantic_check=False)
            if match:
                group = (
                    input_data.filter(
                        (pl.col(make_col).str.to_lowercase() == make.lower())
                        & (pl.col(model_col).str.to_lowercase() == model.lower()),
                    )
                    .drop_nulls(group_col)
                    .group_by(group_col)
                    .agg(pl.count(group_col).alias("count"))
                    .sort("count", descending=True)
                    .to_numpy()
                )
                group = group[0][0] if group.any() else ""
                if len(match) > 1 and group:
                    match_dict = self._check_match(
                        make, model, group=group, use_semantic_check=True,
                    )[0]
                else:
                    match_dict = match[0]
                match_dict["group"] = group
                temp_data.append(match_dict)

        self.aggregated_data = pl.DataFrame(temp_data)

    def _semantic_matching(self, group: str, group_pl: pl.DataFrame) -> dict:
        subcats = group_pl["subcategory"].unique().to_list()
        best_score = -1
        best_subcat = ""
        group_embedding = SENTENCE_MODEL.encode(group, convert_to_tensor=True)
        for subcat in subcats:
            embedding = SENTENCE_MODEL.encode(subcat, convert_to_tensor=True)
            sim_score = util.pytorch_cos_sim(group_embedding, embedding)
            if sim_score > best_score:
                best_score = sim_score
                best_subcat = subcat
        best_fit = group_pl.filter(pl.col("subcategory") == best_subcat)
        return {
            "make": best_fit["make"][0],
            "model": best_fit["model"][0],
            "category": best_fit["category"][0],
            "subcategory": best_fit["subcategory"][0],
        }

    def check_aggregated_data(self, make: str, model: str, group: str) -> dict:
        """Check for a match in the aggregated data.

        Parameters
        ----------
        make : str
            The make of the equipment.
        model : str
            The model of the equipment.
        group : str
            The group of the equipment.

        Returns
        -------
        dict
            The cleaned make, model, category, and subcategory data.

        """
        match = self.aggregated_data.filter(
            (pl.col("make").str.to_lowercase() == make.lower())
            & (pl.col("model").str.to_lowercase() == model.lower()),
        )
        if match.shape[0] > 0:
            return {
                "make": match["make"][0],
                "model": match["model"][0],
                "category": match["category"][0],
                "subcategory": match["subcategory"][0],
            }

        # Assume there are no good matches, get common cat/subcat\
        if group:
            group_category = (
                self.aggregated_data.filter(
                    pl.col("group") == group,
                )
                .group_by("category")
                .agg(pl.count("category").alias("count"))
                .sort("count", descending=True)
                .head(1)
            )["category"][0]
            group_subcategory = (
                self.aggregated_data.filter(
                    pl.col("group") == group,
                )
                .group_by("subcategory")
                .agg(pl.count("subcategory").alias("count"))
                .sort("count", descending=True)
                .head(1)
            )["subcategory"][0]
        else:
            group_category = "Unknown"
            group_subcategory = "Unknown"
        return {
            "make": make,
            "model": model,
            "category": group_category,
            "subcategory": group_subcategory,
        }

    def _check_match(
        self,
        make: str,
        model: str,
        *,
        group: str = "",
        use_semantic_check: bool = False,
    ) -> list:
        """Check for an exact match in the make model data.

        Parameters
        ----------
        make : str
            The make of the equipment.
        model : str
            The model of the equipment.
        group : str
            The dealer designated equipment group. This is used in semantic check to
            identify most similar subcategory.
        use_semantic_check : bool
            Whether to use semantic check to find the best fit subcategory. Group
            must be a non-empty string if this is True

        Returns
        -------
        list
            The cleaned make, model, category, and subcategory data as a list in case
            there are the same make and model but different category and/or subcategory.

        """
        exact_match = self.make_model_data.filter(
            (pl.col("make").str.to_lowercase() == make.lower())
            & (pl.col("model").str.to_lowercase() == model.lower()),
        )
        if use_semantic_check and group and exact_match.shape[0] > 0:
            return [self._semantic_matching(group, exact_match)]

        if exact_match.shape[0] > 0:
            return [
                {
                    "make": make,
                    "model": model,
                    "category": exact_match["category"][i],
                    "subcategory": exact_match["subcategory"][i],
                }
                for i in range(exact_match.shape[0])
            ]
        return [
            {
                "make": make,
                "model": model,
                "category": "Unknown",
                "subcategory": "Unknown",
            },
        ]

    @staticmethod
    def get_make_model_data() -> pl.DataFrame:
        """Fetch make, model, category, and subcategory data from Databricks.

        Returns
        -------
        pl.DataFrame
            The fetched data.

        """
        return read_from_databricks(MAKE_MODEL_QUERY)

    @staticmethod
    def make_synonym_list(acronym: str) -> str:
        """Takes a make acronym and returns a full name"""
        make_acronyms = {
            "JD": "John Deere",
            "SL": "Stihl",
            "CA": "Case IH",
            "XX": "Unknown",
            "UN": "Unverferth",
            "FT": "Frontier",
            "JM": "J&M",
            "FR": "Ferris",
            "CI": "Case IH",
            "BT": "Brent",
            "IH": "International Harvester",
            "HO": "Honda",
            "GV": "Gravely",
            "VT": "Ventrac",
            "HG": "Hagie",
            "MD": "MacDon",
            "ML": "McFarlane",
            "KI": "Kinze",
            "GH": "Geringhoff",
            "HA": "Horsch Anderson",
            "NH": "New Holland",
            "AG": "Agco",
            "CT": "Curtis",
        }

        return make_acronyms.get(acronym, "Unknown")
