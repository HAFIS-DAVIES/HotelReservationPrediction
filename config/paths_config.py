import os

# Data Ingestion paths

RAW_DIR = "artifacts/data_ingestion/raw_data"
RAW_FILE_PATH = os.path.join(RAW_DIR, "Hotel_Reservations.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_FILE_PATH = "config/config.yaml"


# Data Processing paths

PROCESSED_DIR = "artifacts/processed_data"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")

# Model Training paths

MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"
 
   