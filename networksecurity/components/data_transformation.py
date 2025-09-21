import sys
import os
import numpy as np
import pandas as pd

# Scikit-learn for preprocessing
from sklearn.impute import KNNImputer  # for filling missing values using KNN
from sklearn.pipeline import Pipeline  # to create a processing pipeline

# Project constants
from networksecurity.constants.training_pipeline import TARGET_COLUMN  # target column name
from networksecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS  # KNNImputer parameters

# Project entities
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,  # artifact class to store transformed data paths
    DataValidationArtifact  # artifact class for validated data paths
)

from networksecurity.entity.config_entity import DataTransformationConfig  # config for data transformation

# Custom exceptions and logging
from networksecurity.exception.exception import NetworkSecurityException  # custom exception handling
from networksecurity.logging.logger import logging  # logging utility

# Utility functions
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object  # saving arrays and objects


class DataTransformation:
    """
    DataTransformation handles preprocessing of the dataset:
    - Reads validated train/test data.
    - Applies a KNN imputer for missing values.
    - Transforms input features.
    - Saves transformed data and preprocessing objects for model training.
    """

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Initialize the DataTransformation class.

        Args:
            data_validation_artifact (DataValidationArtifact): Holds paths to validated train/test data.
            data_transformation_config (DataTransformationConfig): Holds config for transformed data and objects.
        """
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Read a CSV file and return it as a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(cls) -> Pipeline:
        """
        Create a preprocessing object for data transformation.

        - Initializes a KNNImputer using parameters from training_pipeline.py.
        - Wraps the imputer inside a scikit-learn Pipeline.

        Args:
            cls: Reference to the class (not used, can be ignored).

        Returns:
            Pipeline: Preprocessing pipeline with KNNImputer.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            # Initialize KNNImputer for missing value handling
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialize KNNImputer with params: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            # Create a pipeline with the imputer as the first step
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Perform the full data transformation process:
        1. Load validated train/test datasets.
        2. Separate input features and target column.
        3. Replace -1 in target with 0 for classification.
        4. Fit and apply KNNImputer to input features.
        5. Save transformed datasets as numpy arrays.
        6. Save the preprocessing object for later use in model training.
        7. Return a DataTransformationArtifact with paths.

        Returns:
            DataTransformationArtifact: Paths of transformed train/test data and preprocessor object.
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")

            # Load validated training and testing data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Split input features and target for training data
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)  # Replace -1 with 0

            # Split input features and target for testing data
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)  # Replace -1 with 0

            # Create preprocessing pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit imputer on training input features and transform both train and test
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Combine transformed features with target column
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save transformed data arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            # Save preprocessing object for future use in training/inference
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # Prepare and return artifact containing all file paths
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
