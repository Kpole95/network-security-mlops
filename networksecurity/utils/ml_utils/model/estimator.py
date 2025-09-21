from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    """
    Wrapper class for ML models with a preprocessor.
    Combines preprocessing and model prediction into a single interface.
    """
    def __init__(self, preprocessor, model):
        """
        Initialize NetworkModel with preprocessor and trained model.

        Args:
            preprocessor: A fitted preprocessing object (e.g., Pipeline, KNNImputer)
            model: A trained ML model (e.g., RandomForestClassifier)
        """
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, x):
        """
        Transform input data using preprocessor and make predictions with the model.
        Args:
            x (array-like or DataFrame): Input features for prediction
        Returns:
            y_hat (array-like): Predicted labels
        """
        try:
            # apply preprocessing transformations
            x_transform = self.preprocessor.transform(x)
            # predict using the trained model
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)
