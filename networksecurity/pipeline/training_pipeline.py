import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from networksecurity.constants.training_pipeline import TRAINING_BUCKET_NAME
from networksecurity.cloud.s3_syncer import S3Sync


class TrainingPipeline:
    """
    Orchestrates the complete end-to-end machine learning pipeline for the 
    Network Security project.

    The pipeline includes the following stages:
    1. Data Ingestion
    2. Data Validation
    3. Data Transformation
    4. Model Training
    5. Artifact and model synchronization to AWS S3

    Attributes:
        training_pipeline_config (TrainingPipelineConfig): Configuration for the entire pipeline.
        s3_sync (S3Sync): Utility class to handle syncing folders to S3.
    """

    def __init__(self):
        """
        Initializes the TrainingPipeline with default configuration and S3 sync utility.
        """
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Executes the data ingestion process.

        This method initializes the DataIngestion component using its configuration,
        triggers the ingestion process, and logs the resulting artifact.

        Returns:
            DataIngestionArtifact: Contains information about ingested data,
            such as file paths, schemas, or any metadata generated during ingestion.

        Raises:
            NetworkSecurityException: If any error occurs during the ingestion process.
        """
        try:
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("Start data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Validates the ingested data to ensure quality and correctness.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact returned
                by the data ingestion stage containing ingested data and metadata.

        Returns:
            DataValidationArtifact: Contains validation results, such as
            validation reports, schema checks, and error logs.

        Raises:
            NetworkSecurityException: If validation fails or an unexpected error occurs.
        """
        try:
            config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            validator = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=config
            )
            logging.info("Start data validation")
            return validator.initiate_data_validation()
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Transforms validated data to a suitable format for model training.

        Transformation can include feature engineering, encoding categorical variables,
        normalization, scaling, and splitting data into train/test sets.

        Args:
            data_validation_artifact (DataValidationArtifact): Artifact from
                data validation stage containing clean and validated data.

        Returns:
            DataTransformationArtifact: Artifact containing transformed datasets,
            preprocessing objects (like scalers or encoders), and any metadata.

        Raises:
            NetworkSecurityException: If any error occurs during data transformation.
        """
        try:
            config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            transformer = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=config
            )
            return transformer.initiate_data_transformation()
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Trains the machine learning model using the transformed data.

        This step involves initializing the ModelTrainer component with the
        transformed data and configuration parameters, training the model, and
        generating evaluation metrics.

        Args:
            data_transformation_artifact (DataTransformationArtifact): Artifact
                from the data transformation stage containing preprocessed datasets.

        Returns:
            ModelTrainerArtifact: Contains trained model, evaluation metrics,
            and relevant metadata about training.

        Raises:
            NetworkSecurityException: If model training fails or an unexpected error occurs.
        """
        try:
            self.model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            return trainer.initiate_model_trainer()
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def sync_artifact_dir_to_s3(self):
        """
        Uploads the local artifacts directory to the configured AWS S3 bucket.

        The artifacts directory typically contains logs, intermediate datasets,
        transformation objects, and other outputs from the pipeline.

        Raises:
            NetworkSecurityException: If S3 synchronization fails.
        """
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                aws_bucket_url=aws_bucket_url
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def sync_saved_model_dir_to_s3(self):
        """
        Uploads the local saved model directory to the configured AWS S3 bucket.

        This folder contains the final trained model and any associated files
        necessary for deployment.

        Raises:
            NetworkSecurityException: If S3 synchronization fails.
        """
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.model_dir,
                aws_bucket_url=aws_bucket_url
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def run_pipeline(self) -> ModelTrainerArtifact:
        """
        Executes the full machine learning pipeline end-to-end.

        The steps include:
            1. Data ingestion
            2. Data validation
            3. Data transformation
            4. Model training
            5. Uploading artifacts and final model to S3

        Returns:
            ModelTrainerArtifact: Contains the trained model and evaluation results.

        Raises:
            NetworkSecurityException: If any stage of the pipeline fails.
        
        Example:
            pipeline = TrainingPipeline()
            artifact = pipeline.run_pipeline()
        """
        try:
            # Step 1: Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()

            # Step 2: Data Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)

            # Step 3: Data Transformation
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)

            # Step 4: Model Training
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)

            # Step 5: Sync artifacts and final model to S3
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
