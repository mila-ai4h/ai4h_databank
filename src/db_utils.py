# Setup mongoDB
from dataclasses import dataclass
import os
import logging

from pymongo import MongoClient
from fastapi.encoders import jsonable_encoder

from buster.busterbot import Response

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MONGODB_PASSWORD = os.getenv("MONGODB_AI4H_PASSWORD")
MONGODB_USERNAME = os.getenv("MONGODB_AI4H_USERNAME")
MONGODB_DB_NAME = os.getenv("MONGODB_AI4H_DB_NAME")
ATLAS_URI = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_DB_NAME}.m0zt2w2.mongodb.net/?retryWrites=true&w=majority"


def init_db():
    if all(v is not None for v in [MONGODB_PASSWORD, MONGODB_USERNAME, MONGODB_DB_NAME]):
        mongodb_client = MongoClient(ATLAS_URI)
        database = mongodb_client[MONGODB_DB_NAME]
        logger.info("Connected to the MongoDB database!")

    else:
        database = None
        logger.warning("Didn't connect to MongoDB database, check auth.")

    return database


@dataclass
class Feedback:
    response: Response
    feedback: str
    feedback_extra: str
