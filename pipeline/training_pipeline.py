import os
from src.data_injection import DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTraining
from config.paths_config import *
from utils.common_functions import read_yaml_file
from config.paths_config import *


if __name__ == "__main__":
    ### data injection
    data_ingestion = DataIngestion(read_yaml_file(CONFIG_FILE_PATH))
    data_ingestion.run()
    ### data processing
    processor = DataProcessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_FILE_PATH
    )
    processor.process()

    ### model training
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    trainer.run()
