import os
import sys
import json
import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# load env variables from .env file
from dotenv import load_dotenv
load_dotenv()

# mongodb connection starting form env variable
MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

# SSL certificate varification for MongoDB
import certifi
ca=certifi.where() 


class NetworkDataExtract():
    """
    A class to handle network securiry data extractios and uploads to MONGODB
    """
    def __init__(self):
        """
        constrcted method for networkdataextract,
        currently dos nothing
        """
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    # func to conver csv
    def csv_to_json_convertor(self, file_path):
        """
        Converts CSV file to list of JSON (python dict) records,
        file_path (str): path to csv file.
        returns a List of JSON like dict
        """
        try:
            data=pd.read_csv(file_path) # reads csv file into DF
            data.reset_index(drop=True, inplace=True) # reset index to make sure it is clean for JSon conversion
            records=list(json.loads(data.T.to_json()).values()) # converts DF to JSON string (T for good strcuture)
            return records
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def insert_data_mongodb(self, records, database, collection):
        """
        inserts a list of recods (dicts) into MongoDB collection,

        records (list): A list of json like dicts to inserted,
        database (str): Name of MongpDB database
        collection (str): Name of the MongoDb collection

        returns int: no.of records inserted
        """
        try:

            # saves input args for internal reference
            self.database=database
            self.collection=collection
            self.recods=records

            # connect to MongoDB using url from .env file
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)

            # select the target database and the collection
            self.database=self.mongo_client[self.collection]
            self.collection=self.database[self.collection]

            # insert records into the collection
            self.collection.insert_many(self.recods)
            return (len(self.recods))
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)



if __name__=="__main__":

    # path to the MangoDB collection details
    FILE_PATH="Network_Data\phisingData.csv"
    DATABASE="KRISHNA"
    Collection="NetworkData"

    # create a object
    networkobj=NetworkDataExtract()

    # conver csv to json
    records=networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)

    # insert records into MongoDB
    no_of_records=networkobj.insert_data_mongodb(records, DATABASE, Collection)
    print(no_of_records)