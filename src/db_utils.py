# Setup mongoDB
import logging
from urllib.parse import quote_plus

import pymongo
from pymongo import MongoClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_uri(username: str, password: str, cluster: str) -> str:
    """Create mongodb uri."""
    uri = (
        "mongodb+srv://"
        + quote_plus(username)
        + ":"
        + quote_plus(password)
        + "@"
        + cluster
        + "/?retryWrites=true&w=majority"
    )
    return uri


def init_db(username: str, password: str, cluster: str, db_name: str) -> pymongo.database.Database:
    """Initialize mongodb database."""

    if all(v is not None for v in [username, password, cluster]):
        try:
            uri = make_uri(username, password, cluster)
            mongodb_client = MongoClient(uri)
            database = mongodb_client[db_name]
            logger.info("Succesfully connected to the MongoDB database")
            return database
        except Exception as e:
            logger.exception("Something went wrong connecting to mongodb")

    logger.warning("Didn't connect to MongoDB database, check auth.")
