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

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    """
    Data Ingestion component:
    - Reads data from MongoDB
    - Exports to feature store
    - Splits into train/test datasets
    """
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Read data from MongoDB and return as DataFrame
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            # Connect to MongoDB
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            # Fetch all documents as a list and convert to DataFrame

            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Number of rows fetched from MongoDB: {len(df)}")

            if df.empty:
                raise NetworkSecurityException(
                    error_message="No data found in MongoDB collection.",
                    error_details=sys
                )

            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)
        

            if df.empty:
                logging.warning(f"No data found in collection {collection_name}.")
                return df  # or raise a friendly exception
            logging.info(f"Number of rows fetched from MongoDB: {len(df)}")
            return df

        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Data exported to feature store at {feature_store_file_path}")
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            if dataframe.empty:
                raise NetworkSecurityException(
                    error_message="Cannot split empty DataFrame into train/test.",
                    error_details=sys
                )

            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train/test split on the DataFrame")

            # Ensure directories exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.testing_file_path), exist_ok=True)

            # Save train/test files
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exported train/test files successfully")

        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            df = self.export_collection_as_dataframe()
            df = self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)

            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return artifact

        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)
