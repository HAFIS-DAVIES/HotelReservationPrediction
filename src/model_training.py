import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from config.model_params import *
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import load_data, read_yaml_file
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:

   def __init__(self, train_path, test_path, model_output_path):
       self.train_path = train_path
       self.test_path = test_path
       self.model_output_path = model_output_path

       self.params_dist = LIGHTGM_PARAMS
       self.random_search_params = RANDOM_SEARCH_PARAMS

    
   def load_and_split_data(self):
        """Load and split the data into features and target."""
        try:
            logger.info(f"Loading training and testing data from {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'], errors='ignore')
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=['booking_status'], errors='ignore')
            y_test = test_df['booking_status']

            logger.info("Data splitted successfully for Model Training")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException(f"Failed to loading data: {e}")

   def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing our model")
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info("Starting Randomized Search for Hyperparameter Tuning")
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                random_state=self.random_search_params['random_state'],
                verbose=self.random_search_params['verbose'],
                n_jobs=self.random_search_params['n_jobs'],
                scoring=self.random_search_params['scoring']
            )

            logger.info("Starting model training with Hyperparameter Tuning")
            random_search.fit(X_train, y_train)

            logger.info("Hyperparameter Tuning completed")
            
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best parameters are: {best_params}")

            return best_lgbm_model
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException(f"Failed to train model: {e}")
        
   def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Starting model evaluation")
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Accuracy Score: {accuracy}")
            logger.info(f"Precision Score: {precision}")
            logger.info(f"Recall Score: {recall}")
            logger.info(f"F1 Score: {f1}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException(f"Failed to evaluate model: {e}")
        
   def save_model(self, model):
       try:
           os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

           logger.info(f"Saving the model to {self.model_output_path}")
           joblib.dump(model, self.model_output_path)
           logger.info(f"Model saved successfully at {self.model_output_path}")
       except Exception as e:
           logger.error(f"Error during model saving: {e}")
           raise CustomException(f"Failed to save model: {e}")
    
   def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting Model Training process")

                logger.info("Starting MLflow experimentation")

                logger.info("Logging the training and testing dataset to MLflow")

                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                X_train, y_train, X_test, y_test = self.load_and_split_data()

                best_lgbm_model = self.train_lgbm(X_train, y_train)

                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)

                logger.info("Logging model and metrics to MLflow")

                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging model parameters and metrics to MLflow")
                mlflow.log_params( best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model Training process completed successfully")

        except Exception as e:
            logger.error("Error in Model Training pipeline")
            raise CustomException(f"Model Training process failed: {e}")

if __name__ == "__main__":
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    trainer.run()