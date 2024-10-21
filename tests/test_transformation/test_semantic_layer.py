import json
import pytest

# Load the semantic layer JSON file
def load_json():
    with open('src/transformation/semantic_layer.json') as f:
        return json.load(f)

# Helper function to validate each field within an object
def validate_field(field_data):
    assert 'primary_key' in field_data, "'primary_key' is missing"
    assert 'foreign_key' in field_data, "'foreign_key' is missing"
    assert 'keys' in field_data, "'keys' is missing"
    assert isinstance(field_data['keys'], list), "'keys' should be a list"
    for key_entry in field_data['keys']:
        assert 'org' in key_entry, "'org' is missing in 'keys'"
        assert 'object' in key_entry, "'object' is missing in 'keys'"
        assert 'api_name' in key_entry, "'api_name' is missing in 'keys'"
        assert 'foreign_key_object' in key_entry, "'foreign_key_object' is missing in 'keys'"
        assert 'foreign_key_field' in key_entry, "'foreign_key_field' is missing in 'keys'"

# Test to validate the overall structure of the JSON semantic layer
def test_validate_semantic_layer_structure():
    semantic_layer = load_json()
    
    # Iterate through each top-level object in the semantic layer
    for object_name, object_data in semantic_layer.items():
        assert isinstance(object_data, dict), f"Expected a dictionary for {object_name}"

        # Validate each field within the object
        for field_name, field_data in object_data.items():
            validate_field(field_data)