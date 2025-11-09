import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml_file
from config.paths_config import *

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        self.train_ratio = self.config["train_ratio"]
        
        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(
            f"DataIngestion started with bucket: {self.bucket_name}, "
            f"file: {self.bucket_file_name}, and train ratio: {self.train_ratio}"
        )

    def download_csv_from_gcs(self):
        """Download CSV file from Google Cloud Storage."""
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(
                f"Downloaded raw file: {self.bucket_file_name} from GCS bucket: "
                f"{self.bucket_name} to {RAW_FILE_PATH}"
            )
        except Exception as e:
            logger.error("Error downloading CSV file from GCS")
            raise CustomException(f"Failed to download CSV file from GCS: {e}", sys)

    def split_data(self):
        """Split the data into training and testing sets."""
        try:
            logger.info("Starting data split into train and test sets")
            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(
                data,
                train_size=self.train_ratio,
                random_state=42
            )

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(
                f"Data split completed. Train data saved to {TRAIN_FILE_PATH}, "
                f"Test data saved to {TEST_FILE_PATH}"
            )
        except Exception as e:
            logger.error("Error during data splitting")
            raise CustomException(f"Failed to split data into train and test sets: {e}", sys)

    def run(self):
        """Execute the data ingestion process."""
        try:
            logger.info("Running data ingestion process")
            self.download_csv_from_gcs()
            self.split_data()
            logger.info("Data ingestion process completed successfully")
        except CustomException as ce:
            logger.error("CustomException in data ingestion process")
            logger.error(f"Error details: {ce}")
        finally:
            logger.info("Data ingestion process finished")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml_file(CONFIG_FILE_PATH))
    data_ingestion.run()
