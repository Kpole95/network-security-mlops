from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging 
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    """
    This class checks if our training and testing data are correct
    before we use them for machine learning.

    What it does:
    - Opens the schema file (list of expected columns).
    - Reads train and test CSV files.
    - Checks if all columns are there.
    - Checks if important numerical columns exist.
    - Compares train vs test to see if the data is very different (data drift).
    - Saves the cleaned train and test data.
    """

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Setup for validation.

        - Save where train/test files are.
        - Save where to put validation results.
        - Load schema file (tells us which columns we should expect).
        """
        try:
            # Save train/test file locations from ingestion step
            self.data_ingestion_artifact = data_ingestion_artifact

            # Save output file locations (valid files, drift report, etc.)
            self.data_validation_config = data_validation_config

            # Load schema (expected columns info) from yaml
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Read a CSV file and turn it into a Pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)  # load csv
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Check if the number of columns in data is same as in schema.
        """
        try:
            # Count columns expected in schema
            number_of_columns = len(self._schema_config)

            logging.info(f"Expected columns: {number_of_columns}")
            logging.info(f"Found columns: {len(dataframe.columns)}")

            # Return True if same count, else False
            return len(dataframe.columns) == number_of_columns

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        """
        Compare train and test data to see if they are from the same distribution.

        Uses Kolmogorov-Smirnov test:
        - If p-value > threshold → no drift
        - If p-value < threshold → drift found
        """
        try:
            status = True  # assume no drift
            report = {}    # save drift result here

            # Check every column
            for column in base_df.columns:
                d1, d2 = base_df[column], current_df[column]

                # Compare train and test distributions
                is_same_dist = ks_2samp(d1, d2)

                if threshold <= is_same_dist.pvalue:
                    # distributions look similar
                    is_found = False
                else:
                    # distributions are different → drift
                    is_found = True
                    status = False

                # Save column result
                report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }

            # Save drift report as YAML
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status  # True = no drift, False = drift found

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Check if all required numerical columns are in the data.
        """
        try:
            # Pick numerical columns from schema
            numerical_columns = self._schema_config["numerical_columns"]

            # Check if any are missing
            missing = [col for col in numerical_columns if col not in dataframe.columns]

            if missing:
                logging.error(f"Missing numeric columns: {missing}")
                return False

            logging.info("All numeric columns are present.")
            return True

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Do the full validation process step by step:
        1. Read train and test CSV
        2. Check number of columns
        3. Check numerical columns
        4. Check dataset drift
        5. Save valid train/test CSVs
        6. Return DataValidationArtifact with results
        """
        try:
            # Load train and test datasets
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            # Check train dataset
            if not self.validate_number_of_columns(train_df):
                error_message = "Train file missing columns"
            if not self.is_numerical_column_exist(train_df):
                error_message = "Train file missing numeric columns"

            # Check test dataset
            if not self.validate_number_of_columns(test_df):
                error_message = "Test file missing columns"
            if not self.is_numerical_column_exist(test_df):
                error_message = "Test file missing numeric columns"

            # Check for dataset drift
            status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            # Save valid train and test CSVs
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            # Return the results
            return DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)