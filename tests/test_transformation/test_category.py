import polars as pl
import pytest

from src.transformation.category import CleanMakeModelData


@pytest.fixture
def clean_make_model_data():
    # Initialize the CleanMakeModelData instance
    data_instance = CleanMakeModelData()
    return data_instance


@pytest.fixture
def sample_make_model_data():
    # Sample data similar to what would be fetched
    return pl.DataFrame(
        {
            "make": ["John Deere", "Stihl", "Case IH"],
            "model": ["X300", "MS180", "Puma"],
            "category": ["Tractor", "Chainsaw", "Tractor"],
            "subcategory": ["Lawn", "Handheld", "Agriculture"],
        },
    )


@pytest.mark.parametrize(
    "make, model, expected",
    [
        (
            "John Deere",
            "X300",
            {
                "make": "John Deere",
                "model": "X300",
                "category": "Tractor",
                "subcategory": "Lawn",
            },
        ),
        (
            "Stihl",
            "MS180",
            {
                "make": "Stihl",
                "model": "MS180",
                "category": "Chainsaw",
                "subcategory": "Handheld",
            },
        ),
        (
            "Unknown",
            "X300",
            {
                "make": "Unknown",
                "model": "X300",
                "category": "Unknown",
                "subcategory": "Unknown",
            },
        ),
    ],
)
def test_01_clean_make_model_data(
    clean_make_model_data, sample_make_model_data, make, model, expected,
):
    # Assign mock data to the instance
    clean_make_model_data.make_model_data = sample_make_model_data
    result = clean_make_model_data.clean_make_model_data(make, model)
    assert result == expected


def test_02_create_aggregated_data(clean_make_model_data):
    input_data = pl.DataFrame(
        {
            "dsu_make": ["John Deere", "Stihl", "Case IH"],
            "dsu_model": ["X300", "MS180", "Puma"],
            "dsu_group": ["Lawn", "Handheld", "Agriculture"],
        },
    )
    clean_make_model_data.create_aggregated_data(input_data)
    assert not clean_make_model_data.aggregated_data.is_empty()


@pytest.mark.parametrize(
    "make, model, group, expected",
    [
        (
            "John Deere",
            "X300",
            "Lawn",
            {
                "make": "John Deere",
                "model": "X300",
                "category": "Tractor",
                "subcategory": "Lawn",
            },
        ),
        (
            "Stihl",
            "MS180",
            "Handheld",
            {
                "make": "Stihl",
                "model": "MS180",
                "category": "Chainsaw",
                "subcategory": "Handheld",
            },
        ),
    ],
)
def test_03_check_aggregated_data(
    clean_make_model_data, sample_make_model_data, make, model, group, expected,
):
    clean_make_model_data.aggregated_data = sample_make_model_data
    result = clean_make_model_data.check_aggregated_data(make, model, group)
    assert result == expected


def test_04_semantic_matching(clean_make_model_data):
    group = "Agriculture"
    group_pl = pl.DataFrame(
        {
            "make": ["Case IH", "John Deere"],
            "model": ["Puma", "X300"],
            "category": ["Tractor", "Tractor"],
            "subcategory": ["Agriculture", "Lawn"],
        },
    )
    result = clean_make_model_data._semantic_matching(group, group_pl)
    assert result["subcategory"] == "Agriculture"


@pytest.mark.parametrize(
    "acronym, expected",
    [("JD", "John Deere"), ("SL", "Stihl"), ("XX", "Unknown"), ("ZZ", "Unknown")],
)
def test_05_make_synonym_list(clean_make_model_data, acronym, expected):
    result = clean_make_model_data.make_synonym_list(acronym)
    assert result == expected


def test_06_get_make_model_data(mocker):
    mock_read = mocker.patch("src.utils.io.read_from_databricks")
    mock_read.return_value = pl.DataFrame(
        {
            "make": ["John Deere"],
            "model": ["X300"],
            "category": ["Tractor"],
            "subcategory": ["Lawn"],
        },
    )
    result = CleanMakeModelData.get_make_model_data()
    assert not result.is_empty()


@pytest.mark.parametrize(
    "make, model, group, use_semantic_check, expected",
    [
        (
            "John Deere",
            "X300",
            "",
            False,
            [
                {
                    "make": "John Deere",
                    "model": "X300",
                    "category": "Tractor",
                    "subcategory": "Lawn",
                },
            ],
        ),
        (
            "Unknown",
            "X300",
            "",
            False,
            [
                {
                    "make": "Unknown",
                    "model": "X300",
                    "category": "Unknown",
                    "subcategory": "Unknown",
                },
            ],
        ),
    ],
)
def test_07_check_match(
    clean_make_model_data,
    sample_make_model_data,
    make,
    model,
    group,
    use_semantic_check,
    expected,
):
    clean_make_model_data.make_model_data = sample_make_model_data
    result = clean_make_model_data._check_match(
        make, model, group=group, use_semantic_check=use_semantic_check,
    )
    assert result == expected
