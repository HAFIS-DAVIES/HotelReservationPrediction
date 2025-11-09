import os
import sys
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import load_data, read_yaml_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, train_path, test_path, config_path, processed_dir):
        self.train_path = train_path
        self.test_path = test_path
        self.config = read_yaml_file(config_path)
        self.processed_dir = processed_dir
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self,df):
        """Preprocess the data: handle missing values, encode categorical variables, etc."""
        try:
            logger.info("Starting data preprocessing")
            
            logger.info("Dropping rows with missing target values")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True, errors='ignore')
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Applying Label Encoding to categorical columns")
            label_encoders = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoders.fit_transform(df[col])
                mappings[col] = dict(zip(label_encoders.classes_, label_encoders.transform(label_encoders.classes_)))    

            logger.info("Label Mappings:")
            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            logger.info("Doing Skewness Correction using SMOTE")  
            skewness_threshold = self.config["data_processing"]["skewness_threshold"]   
            skewness = df[num_cols].apply(lambda x: x.skew())

            for column in skewness[skewness>skewness_threshold].index:
                df[column] = np.log1p(df[column])

            return df    

        except Exception as e:
            logger.error("Error during data preprocessing")
            raise CustomException(f"Data preprocessing failed: {e}")


    def balance_data(self, df):
        """Balance the dataset """
        try:
            logger.info("Handling imbalanced data")
            X = df.drop(columns='booking_status')
            y = df['booking_status']

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status'] = y_resampled

            logger.info("Data balancing completed")
            return balanced_df
        
        except Exception as e:
            logger.error("Error during data balancing")
            raise CustomException(f"Data balancing failed: {e}")

    def select_features(self, df):
        """Select important features based on configuration."""
        try:
            logger.info("Starting feature selection")
            X = df.drop(columns='booking_status')
            y = df['booking_status']

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importances
            })

            top_features_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            num_features_to_select = self.config["data_processing"]["no_of_features"]

            top_10_features = top_features_importance_df['Feature'].head(num_features_to_select).values

            logger.info(f"Top features selected: {top_10_features}")

            top_10_df = df[top_10_features.tolist() + ['booking_status']]

            logger.info("Feature selection completed")

            return top_10_df
        
        except Exception as e:
            logger.error("Error during feature selection")
            raise CustomException(f"Feature selection failed: {e}")

    def save_data(self,df, file_path):
        """Save the processed data to a CSV file."""
        try:
            logger.info(f"Saving processed data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to {file_path}")

        except Exception as e:
            logger.error("Error saving processed data")
            raise CustomException(f"Failed to save processed data: {e}")
        
    def process(self):
        """Execute the data processing pipeline."""
        try:
            logger.info("Loading data from RAW directory")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            logger.info("Preprocessing training data")
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            logger.info("Balancing training data")
            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            logger.info("Selecting features from training data")
            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing pipeline completed successfully")

        except CustomException as ce:
            logger.error("CustomException in data processing pipeline")
            logger.error(f"Error while processing data: {ce}")

            
if __name__ == "__main__":
    processor = DataProcessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_FILE_PATH
    )
    processor.process()