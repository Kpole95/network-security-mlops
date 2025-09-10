import pymongo
import os
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
client = pymongo.MongoClient(MONGO_DB_URL)

db = client["NetworkData"]
collection = db["NetworkData"]

docs = list(collection.find())
print("Fetched rows:", len(docs))  # Should print 11055
print("First row keys:", docs[0].keys() if docs else "Empty")
