# Setup mongoDB
import logging
import os
from dataclasses import dataclass
from urllib.parse import quote_plus

from buster.busterbot import Response
from fastapi.encoders import jsonable_encoder
from pymongo import MongoClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def init_db():
    """Initialize mongodb database."""

    username = os.getenv("AI4H_MONGODB_USERNAME")
    password = os.getenv("AI4H_MONGODB_PASSWORD")
    cluster = os.getenv("AI4H_MONGODB_CLUSTER")
    db_name = os.getenv("AI4H_MONGODB_DB_NAME")

    if all(v is not None for v in [username, password, cluster]):
        try:
            uri = (
                "mongodb+srv://"
                + quote_plus(username)
                + ":"
                + quote_plus(password)
                + "@"
                + cluster
                + "/?retryWrites=true&w=majority"
            )
            mongodb_client = MongoClient(uri)
            database = mongodb_client[db_name]
            logger.info("Succesfully connected to the MongoDB database")
            return database
        except Exception as e:
            logger.exception("Something went wrong connecting to mongodb")

    logger.warning("Didn't connect to MongoDB database, check auth.")


@dataclass
class Feedback:
    good_bad: str
    extra_info: str
    relevant_answer: str
    relevant_length: str
    relevant_sources: str
    length_sources: str
    timeliness_sources: str
