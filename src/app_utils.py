import logging
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import quote_plus

import pandas as pd
import pymongo
from pymongo import MongoClient

from buster.tokenizers import Tokenizer

# auth information
USERNAME = os.environ["AI4H_APP_USERNAME"]
PASSWORD = os.environ["AI4H_APP_PASSWORD"]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WordTokenizer(Tokenizer):
    """Naive word-level tokenizer

    The original tokenizer from openAI eats way too much Ram.
    This is a naive word count tokenizer to be used instead."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, string):
        return string.split()

    def decode(self, encoded):
        return " ".join(encoded)


def get_session_id() -> str:
    """Generate a uuid for each user."""
    return str(uuid.uuid1())


def verify_required_env_vars(required_vars: list[str]):
    unset_vars = [var for var in required_vars if os.getenv(var) is None]
    if len(unset_vars) > 0:
        logger.warning(f"Lisf of env. variables that weren't set: {unset_vars}")
    else:
        logger.info("All environment variables are set appropriately.")


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


def get_utc_time() -> str:
    return str(datetime.now(timezone.utc))


def check_auth(username: str, password: str) -> bool:
    """Check if authentication succeeds or not.

    The authentication leverages the built-in gradio authentication. We use a shared password among users.
    It is temporary for developing the PoC. Proper authentication needs to be implemented in the future.
    We allow a valid username to be any username beginning with 'databank-', this will allow us to differentiate between users easily.
    """
    valid_user = username.startswith(USERNAME)
    valid_password = password == PASSWORD
    is_auth = valid_user and valid_password
    logger.info(f"Log-in attempted by {username=}. {is_auth=}")
    return is_auth


def format_sources(matched_documents: pd.DataFrame) -> list[str]:
    formatted_sources = []

    for _, doc in matched_documents.iterrows():
        formatted_sources.append(f"### [{doc.title}]({doc.url})\n{doc.content}\n")

    return formatted_sources


def pad_sources(sources: list[str], max_sources: int) -> list[str]:
    """Pad sources with empty strings to ensure that the number of sources is always max_sources."""
    k = len(sources)
    return sources + [""] * (max_sources - k)


def add_sources(completion, max_sources: int):
    if any(arg is False for arg in [completion.question_relevant, completion.answer_relevant]):
        # Question was not relevant, don't bother doing anything else...
        formatted_sources = [""]
        formatted_sources = pad_sources(formatted_sources, max_sources)
        return formatted_sources

    formatted_sources = format_sources(completion.matched_documents)

    return formatted_sources
