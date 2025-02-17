"""
Module for handling JSON data loading and processing.
"""
import json

def load_data_from_json(filepath):
    """Load and return JSON data from a file."""
    with open(filepath, "r") as f:
        data_json = json.load(f)
    return data_json
