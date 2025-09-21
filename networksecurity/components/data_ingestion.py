from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load MongoDB URL from environment variables
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")  # MongoDB connection URL


class DataIngestion:
    """
    Handles data ingestion from MongoDB to the local file system.
    
    Steps:
    1. Fetch data from a MongoDB collection.
    2. Save the data to a feature store (CSV file).
    3. Split the data into training and testing datasets.
    4. Return paths of the saved train/test files as a DataIngestionArtifact.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize the DataIngestion object with the given configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration object containing
                database name, collection name, file paths, and train-test split ratio.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Connects to MongoDB, fetches all documents from the specified collection, 
        converts them into a pandas DataFrame, and handles basic cleaning.
        
        Returns:
            pd.DataFrame: Data fetched from MongoDB.
        
        Raises:
            NetworkSecurityException: If fetching fails or collection is empty.
        """
        try:
            db_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            # Connect to MongoDB and fetch collection data
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[db_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            logging.info(f"Fetched {len(df)} rows from MongoDB collection {collection_name}")

            if df.empty:
                raise NetworkSecurityException(
                    error_message="No data found in MongoDB collection.",
                    error_details=sys
                )

            # Drop MongoDB default '_id' column if exists
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)

            # Replace string 'na' with np.nan for consistency
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Saves the DataFrame into a feature store CSV file as specified in the configuration.
        
        Args:
            dataframe (pd.DataFrame): Data to save.
        
        Returns:
            pd.DataFrame: The same DataFrame after saving.
        """
        try:
            file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            dataframe.to_csv(file_path, index=False, header=True)
            logging.info(f"Data exported to feature store at {file_path}")
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the given DataFrame into training and testing sets based on the 
        configured ratio and saves them to the configured file paths.
        
        Args:
            dataframe (pd.DataFrame): The full dataset to split.
        
        Raises:
            NetworkSecurityException: If DataFrame is empty or saving fails.
        """
        try:
            if dataframe.empty:
                raise NetworkSecurityException(
                    error_message="Cannot split empty DataFrame into train/test.",
                    error_details=sys
                )

            # Split the data
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train/test split")

            # Ensure directories exist and save the splits
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.testing_file_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Saved train/test datasets successfully")

        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Runs the full data ingestion pipeline:
        1. Fetch data from MongoDB.
        2. Save to feature store.
        3. Split into train/test sets.
        4. Return a DataIngestionArtifact containing file paths of train and test datasets.
        
        Returns:
            DataIngestionArtifact: Contains paths of training and testing CSV files.
        """
        try:
            df = self.export_collection_as_dataframe()  # Step 1: fetch
            df = self.export_data_into_feature_store(df)  # Step 2: save
            self.split_data_as_train_test(df)  # Step 3: split

            # Step 4: create artifact
            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return artifact

        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)