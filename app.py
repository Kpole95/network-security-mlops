import sys
import os
import certifi
from dotenv import load_dotenv
import pymongo
import pandas as pd

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constants.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# Load environment variables
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print(f"MongoDB URL loaded: {mongo_db_url}")

# Certifi CA file for TLS connection
ca = certifi.where()

# Initialize MongoDB client
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Initialize FastAPI app
app = FastAPI()

# CORS settings: allow all origins for simplicity
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Jinja2 templates dir
templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authentication"])
async def index():
    """
    Root route that redirects to API documentation.

    Returns:
        RedirectResponse: Redirects user to "/docs".
    """
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["training"])
async def train_route():
    """
    Route to trigger the end-to-end training pipeline.

    This route executes the entire TrainingPipeline:
        1. Data ingestion
        2. Data validation
        3. Data transformation
        4. Model training
        5. Sync artifacts and final model to S3

    Returns:
        Response: Success message if training completes.

    Raises:
        NetworkSecurityException: If any step of the training pipeline fails.
    """
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict", tags=["prediction"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    """
    Route to perform prediction using the trained model.

    Steps:
        1. Read the uploaded CSV file into a pandas DataFrame.
        2. Load preprocessor and trained model objects.
        3. Create a NetworkModel instance with preprocessor and model.
        4. Predict the target column.
        5. Append predictions to the DataFrame.
        6. Save predictions to CSV and render as HTML table.

    Args:
        request (Request): FastAPI request object.
        file (UploadFile): CSV file uploaded by the user.

    Returns:
        TemplateResponse: Renders HTML table with predictions.

    Raises:
        NetworkSecurityException: If file reading, preprocessing, or prediction fails.
    """
    try:
        df=pd.read_csv(file.file)
        #print(df)
        preprocesor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocesor,model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
            raise NetworkSecurityException(e,sys)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
