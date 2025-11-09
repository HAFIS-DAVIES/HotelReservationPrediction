import os
import yaml
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def read_yaml_file(file_path: str):
    """Safely read and parse a YAML configuration file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

        logger.info(f"YAML file '{file_path}' read successfully.")
        return config

    except FileNotFoundError as e:
        logger.error(f"YAML file not found: {file_path}")
        raise CustomException(e)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML syntax in file: {file_path}")
        raise CustomException(e)
    except Exception as e:
        logger.error(f"Unexpected error while reading YAML file: {file_path}")
        raise CustomException(e)

def load_data(path):
    """Load data from a CSV file into a pandas DataFrame."""
    try:
       logger.info(f"Loading data from {path}")
       return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading CSV file {path}: {e}")
        raise CustomException("Failed to load data", e)
