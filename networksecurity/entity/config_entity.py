import networksecurity.constants.training_pipeline as tp
print("TRAINING PIPELINE FILE:", tp.__file__)


from datetime import datetime
import os
from networksecurity.constants import training_pipeline


print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)


class TrainingPipelineConfig:
    """
    Configuration class for the entire training pipeline.
    
    Attributes:
        pipeline_name (str): Name of the pipeline.
        artifact_name (str): Base directory for storing artifacts.
        artifact_dir (str): Timestamped directory for this pipeline run.
        timestamp (str): Timestamp string used in artifact_dir.
    """
    def __init__(self, timestamp=datetime.now()):
        # timestamp format
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str =training_pipeline.PIPELINE_NAME
        self.artifact_name: str =training_pipeline.ARTIFACT_DIR

        # timestamp dir for this pipeline runs artifacts
        self.artifact_dir: str =os.path.join(self.artifact_name, timestamp)
        self.timestamp: str=timestamp


class DataIngestionConfig:
    """
    Configuration for Data Ingestion component.
    
    Attributes:
        data_ingestion_dir (str): Base directory for data ingestion artifacts.
        feature_store_file_path (str): Path to save raw/full dataset.
        training_file_path (str): Path to save training dataset.
        testing_file_path (str): Path to save testing dataset.
        train_test_split_ratio (float): Ratio for splitting train/test datasets.
        collection_name (str): MongoDB collection name to read data from.
        database_name (str): MongoDB database name.
    """

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        # base dir for all the data ingestion artifacts
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME
        )
        # path to store full, raw dataset
        self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
            )
        # path for final training data file
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME
            )
        # path for final testing data file
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
            )
         
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    """
    Configuration for Data Validation component.
    
    Attributes:
        data_validation_dir (str): Base directory for validation artifacts.
        valid_data_dir (str): Directory to store valid data.
        invalid_data_dir (str): Directory to store invalid data.
        valid_train_file_path (str): Path to validated training data.
        valid_test_file_path (str): Path to validated test data.
        invalid_train_file_path (str): Path to invalid training data.
        invalid_test_file_path (str): Path to invalid test data.
        drift_report_file_path (str): Path to store dataset drift report.
    """
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join( training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )

class DataTransformationConfig:
     """
    Configuration for Data Transformation component.
    
    Attributes:
        data_transformation_dir (str): Base directory for data transformation artifacts.
        transformed_train_file_path (str): Path to transformed training dataset (.npy).
        transformed_test_file_path (str): Path to transformed testing dataset (.npy).
        transformed_object_file_path (str): Path to save preprocessing object (e.g., KNN imputer pipeline).
    """
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"), )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,)
        
class ModelTrainerConfig:
    """
    Configuration for Model Trainer component.
    
    Attributes:
        model_trainer_dir (str): Base directory for model trainer artifacts.
        trained_model_file_path (str): Path to save trained model.
        expected_accuracy (float): Minimum expected model accuracy.
        overfitting_underfitting_threshold (float): Allowed tolerance between train/test metrics.
    """
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD