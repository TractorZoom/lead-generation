"""Contains functionality to map Anvil equipment data to TZ Cat + Subcat."""

import logging

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer, util

from src.utils.io import read_from_databricks

SENTENCE_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

MAX_CHARACTERS_IN_ACROYNM = 3
MIN_SIMILARITY_DEVIATION = 0.02
BEST_SCORE_THRESHOLD = 0.5

MAKE_MODEL_QUERY = """SELECT m2.id
     , m1.name AS make
     , m2.model
     , cat.name AS category
     , sub.name AS subcategory
     , COUNT(ed.id) AS total_units
FROM prod_mysql.silver_tractor_prod.make m1
JOIN prod_mysql.silver_tractor_prod.make_model m2 ON m1.id = m2.make_id
JOIN prod_mysql.silver_tractor_prod.lot_category cat ON cat.id = m2.category
JOIN prod_mysql.silver_tractor_prod.lot_subcategory sub ON sub.id = m2.subcategory
JOIN prod_mysql.silver_tractor_prod.equipment_details ed ON ed.make_model_id = m2.id
WHERE m2.status = 1
GROUP BY ALL
ORDER BY total_units DESC"""

log = logging.getLogger(__name__)


class CleanMakeModelData:
    """Class to clean make model data and map to TZ Cat + Subcat."""

    def __init__(self) -> None:
        """Initialize the CleanMakeModelData class."""
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
        group : str
            The equipment group defined by the dealer. If this isn't filled out
            it can't use the semantic check to find the best subcategory.

        Returns
        -------
        dict
            The cleaned make, model, category, and subcategory data.

        """
        # check for exact match in make model data
        exact_match = self._check_match(
            make,
            model,
            group=group,
            use_semantic_check=True,
        )
        synonym_make = None
        if exact_match and exact_match[0]["best_fit_reason"] == "Exact Match":
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
            if match_check and match_check[0]["best_fit_reason"] in [
                "Acronym",
                "Exact Match",
            ]:
                return match_check[0]

        # check for most most likely based on aggregated data
        if self.aggregated_data.shape[0] > 0:
            aggregated_check = self.check_aggregated_data(make, model, group)
            if aggregated_check and aggregated_check["category"] != "Unknown":
                return aggregated_check

        # check for semantic match
        if group:
            semantic_check = self._check_match(
                synonym_make if synonym_make else make,
                model,
                group=group,
                use_semantic_check=True,
            )
            if semantic_check and semantic_check[0]["best_fit_reason"].startswith(
                "Semantic"
            ):
                return semantic_check[0]

        check_no_special_chars = self.check_with_no_special_characters(
            make,
            model,
            group,
        )
        if check_no_special_chars and (
            check_no_special_chars["best_fit_reason"] != "No Match"
        ):
            return check_no_special_chars

        # check for best guess based on group
        if group:
            best_guess = self.get_best_guess_cat_subcat(
                synonym_make if synonym_make else make, model, group
            )
            if best_guess and best_guess["category"] != "Unknown":
                return best_guess

        return exact_match[0]

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
                        make,
                        model,
                        group=group,
                        use_semantic_check=True,
                    )[0]
                else:
                    match_dict = match[0]
                match_dict["group"] = group
                temp_data.append(match_dict)

        self.aggregated_data = pl.DataFrame(temp_data)

    def _semantic_matching(self, group: str, group_pl: pl.DataFrame) -> dict:
        subcats = (
            group_pl.with_columns(
                (pl.col("category") + "-" + pl.col("subcategory")).alias("combined")
            )
            .select("combined")
            .unique()
            .to_series()
            .to_list()
        )
        best_score = -1
        best_subcat = ""
        best_fit_reason = ""
        scores = []
        group_embedding = SENTENCE_MODEL.encode(
            group, convert_to_tensor=True, show_progress_bar=False
        )
        for subcat in subcats:
            embedding = SENTENCE_MODEL.encode(
                subcat, convert_to_tensor=True, show_progress_bar=False
            )
            sim_score = util.pytorch_cos_sim(group_embedding, embedding)
            scores.append(sim_score.item())
            if sim_score > best_score:
                best_score = sim_score.item()
                best_subcat = "-".join(subcat.split("-")[1:])
                best_fit_reason = "Semantic - Best Match"

        if best_score < BEST_SCORE_THRESHOLD:
            best_subcat = group_pl["subcategory"][0]
            best_fit_reason = "Semantic - Most Frequent (No Good Match)"

        # check if best score is very similar to the other scores
        try:
            second_best_score = sorted(scores)[-2]
        except (
            IndexError
        ):  # Case where the same equipment is listed twice in a subcategory
            second_best_score = 1
        if best_score - second_best_score < MIN_SIMILARITY_DEVIATION:
            best_subcat = group_pl["subcategory"][0]
            best_fit_reason = "Semantic - Most Frequent (No Clear Best Match)"

        best_fit = group_pl.filter(pl.col("subcategory") == best_subcat)

        return {
            "make": best_fit["make"][0],
            "model": best_fit["model"][0],
            "category": best_fit["category"][0],
            "subcategory": best_fit["subcategory"][0],
            "best_fit_reason": best_fit_reason,
            "best_fit_score": best_score,
        }

    def get_best_guess_cat_subcat(self, make: str, model: str, group: str) -> dict:
        """Get the most common category and subcategory for a group.

        Parameters
        ----------
        make: str
            The make of the equipment.
        model: str
            The model of the equipment.
        group : str
            The group of the equipment.

        Returns
        -------
        dict
            The most common category and subcategory for the group.

        """
        group_category = (
            self.aggregated_data.filter(
                pl.col("group") == group,
            )
            .filter(pl.col("category") != "Unknown")
            .group_by("category")
            .agg(pl.count("category").alias("count"))
            .sort("count", descending=True)
            .head(1)
        )["category"]
        group_subcategory = (
            self.aggregated_data.filter(
                pl.col("group") == group,
            )
            .filter(pl.col("subcategory") != "Unknown")
            .group_by("subcategory")
            .agg(pl.count("subcategory").alias("count"))
            .sort("count", descending=True)
            .head(1)
        )["subcategory"]

        if not group_category.is_empty() and not group_subcategory.is_empty():
            return {
                "make": make,
                "model": model,
                "category": group_category[0],
                "subcategory": group_subcategory[0],
                "best_fit_reason": "No Match - Estimate Cat/Subcat",
                "best_fit_score": -1,
            }
        return {
            "make": make,
            "model": model,
            "category": "Unknown",
            "subcategory": "Unknown",
            "best_fit_reason": "No Match",
            "best_fit_score": -1,
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
                "best_fit_reason": "Exact Match",
                "best_fit_score": 1,
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
            )["category"]
            group_subcategory = (
                self.aggregated_data.filter(
                    pl.col("group") == group,
                )
                .group_by("subcategory")
                .agg(pl.count("subcategory").alias("count"))
                .sort("count", descending=True)
                .head(1)
            )["subcategory"]

        else:
            # create empty series
            group_category = pl.Series()
            group_subcategory = pl.Series()

        if not group_category.is_empty() and not group_subcategory.is_empty():
            return {
                "make": make,
                "model": model,
                "category": group_category[0],
                "subcategory": group_subcategory[0],
                "best_fit_reason": "Aggregated Most Likely",
                "best_fit_score": -1,
            }
        return {
            "make": make,
            "model": model,
            "category": "Unknown",
            "subcategory": "Unknown",
            "best_fit_reason": "No Match",
            "best_fit_score": -1,
        }

    def _check_match(
        self,
        make: str,
        model: str,
        best_match_reason: str = "",
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
        best_match_reason : str
            The reason for the best match. This is used in the output.
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
        # check for acronym in make
        if len(make) <= MAX_CHARACTERS_IN_ACROYNM:
            synonym_make = self.make_synonym_list(make)
            return self._check_match(
                synonym_make,
                model,
                group=group,
                use_semantic_check=True,
                best_match_reason="Acronym",
            )

        if use_semantic_check and group and exact_match.shape[0] > 1:
            return [self._semantic_matching(group, exact_match)]

        if exact_match.shape[0] > 0:
            if not best_match_reason:
                best_match_reason = "Exact Match"
            return [
                {
                    "make": make,
                    "model": model,
                    "category": exact_match["category"][i],
                    "subcategory": exact_match["subcategory"][i],
                    "best_fit_reason": best_match_reason,
                    "best_fit_score": 1,
                }
                for i in range(exact_match.shape[0])
            ]
        return [
            {
                "make": make,
                "model": model,
                "category": "Unknown",
                "subcategory": "Unknown",
                "best_fit_reason": "No Match",
                "best_fit_score": -1,
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
        """Turn an acronym into a full name.

        Parameters
        ----------
        acronym : str
            The acronym to convert to a full name.

        Returns
        -------
        str
            The full name of the acronym.

        """
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

    def check_with_no_special_characters(
        self, make: str, model: str, group: str = ""
    ) -> dict:
        """Check for a match with no special characters in the make.

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
            The cleaned make, model, category, and subcategory data
        """
        # check for exact match by removing special characters
        make_no_special_chars = "".join(e for e in make if e.isalnum())
        exact_match = self._check_match(
            make_no_special_chars,
            model,
            group=group,
            use_semantic_check=True,
        )
        if exact_match and exact_match[0]["best_fit_reason"] == "Exact Match":
            return exact_match[0]
        # check for semantic match without special characters
        if group:
            semantic_check = self._check_match(
                make_no_special_chars,
                model,
                group=group,
                use_semantic_check=True,
            )
            if semantic_check and semantic_check[0]["best_fit_reason"].startswith(
                "Semantic"
            ):
                return semantic_check[0]
        return exact_match[0]
