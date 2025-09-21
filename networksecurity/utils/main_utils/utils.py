import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
#import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    
    Args:
        file_path (str): Path to the YAML file.
    
    Returns:
        dict: Content of the YAML file.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes a Python object to a YAML file.
    
    Args:
        file_path (str): Path where the YAML file will be written.
        content (object): Python object to serialize to YAML.
        replace (bool): If True, replaces existing file.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file.
    Args:
        file_path (str): Path to save the numpy array.
        array (np.array): Numpy array to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    

    
def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using pickle.
    
    Args:
        file_path (str): Path to save the object.
        obj (object): Python object to serialize.
    """
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
    
def load_object(file_path: str, ) -> object:
    """
    Load a Python object from a pickle file.
    Args:
        file_path (str): Path to the pickle file.
    Returns:
        object: Loaded Python object.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array data from file
    Args:
        file_path (str): Path to the numpy file (.npy).
    Returns:
        np.array: Loaded numpy array.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    """
    Evaluate multiple ML models with hyperparameter tuning using GridSearchCV.
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_test (np.array): Test features.
        y_test (np.array): Test labels.
        models (dict): Dictionary of model_name -> model_object.
        param (dict): Dictionary of model_name -> hyperparameter grid.
    
    Returns:
        dict: Dictionary of model_name -> test R2 score.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)