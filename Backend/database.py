from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Loads variables from a .env file sitting next to this file (Backend/.env)
# so you don't have to set MONGO_URI manually in every terminal session.
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "stress_detector")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]


def get_db():
    """FastAPI dependency — yields the Mongo database handle."""
    return db