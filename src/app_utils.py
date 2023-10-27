import logging
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import quote_plus

import gradio as gr
import pandas as pd
import pymongo
from pymongo import MongoClient

from buster.tokenizers import Tokenizer

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


def get_logging_db_name(instance_type: str) -> str:
    assert instance_type in ["dev", "prod", "local", "test"], "Invalid instance_type declared."
    return f"ai4h-databank-{instance_type}"


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


def init_db(mongo_uri: str, db_name: str) -> pymongo.database.Database:
    """
    Initialize and return a connection to the specified MongoDB database.

    Parameters:
    - mongo_uri (str): The connection string for the MongoDB. This can be formed using `make_uri` function.
    - db_name (str): The name of the MongoDB database to connect to.

    Returns:
    pymongo.database.Database: The connected database object.

    Note:
    If there's a problem with the connection, an exception will be logged and the program will terminate.
    """

    try:
        mongodb_client = MongoClient(mongo_uri)
        # Ping the database to make sure authentication is good
        mongodb_client.admin.command("ping")
        database = mongodb_client[db_name]
        logger.info("Succesfully connected to the MongoDB database")
        return database
    except Exception as e:
        logger.exception("Something went wrong connecting to mongodb")


def get_utc_time() -> str:
    return str(datetime.now(timezone.utc))


def check_auth(username: str, password: str) -> bool:
    """Check if authentication succeeds or not.

    The authentication leverages the built-in gradio authentication. We use a shared password among users.
    It is temporary for developing the PoC. Proper authentication needs to be implemented in the future.
    We allow a valid username to be any username beginning with 'databank-', this will allow us to differentiate between users easily.
    """

    # get auth information from env. vars, they need to be set
    USERNAME = os.environ["AI4H_APP_USERNAME"]
    PASSWORD = os.environ["AI4H_APP_PASSWORD"]

    valid_user = username.startswith(USERNAME)
    valid_password = password == PASSWORD
    is_auth = valid_user and valid_password
    logger.info(f"Log-in attempted by {username=}. {is_auth=}")
    return is_auth


def format_sources(matched_documents: pd.DataFrame) -> list[str]:
    formatted_sources = []

    # We first group on Title of the document, so that 2 chunks from a same doc get lumped together
    grouped_df = matched_documents.groupby("title")

    # Here we just rank the titles by highest to lowest similarity score...
    ranked_titles = (
        grouped_df.apply(lambda x: x.similarity_to_answer.max()).sort_values(ascending=False).index.to_list()
    )

    for title in ranked_titles:
        df = grouped_df.get_group(title)

        # Adds a link break between sources from a same chunk
        chunks = "<br><br>".join(["🔗 " + chunk for chunk in df.content.to_list()])

        url = df.url.to_list()[0]
        source = df.source.to_list()[0]
        year = df.year.to_list()[0]
        country = df.country.to_list()[0]

        formatted_sources.append(
            f"""

### Publication: [{title}]({url})
**Year of publication:** {year}
**Source:** {source}
**Country:** {country}

**Identified sections**:
{chunks}
"""
        )

    return formatted_sources


def pad_sources(sources: list[str], max_sources: int) -> list[str]:
    """Pad sources with empty strings to ensure that the number of sources is always max_sources."""
    k = len(sources)
    return sources + [""] * (max_sources - k)


def add_sources(completion, max_sources: int):
    if any(arg is False for arg in [completion.question_relevant, completion.answer_relevant]):
        # Question was not relevant, don't bother doing anything else...
        formatted_sources = [""]
    else:
        formatted_sources = format_sources(completion.matched_documents)

    formatted_sources = pad_sources(formatted_sources, max_sources)

    sources_textboxes = []
    for source in formatted_sources:
        visible = False if source == "" else True
        t = gr.Markdown(source, latex_delimiters=[], elem_classes="source", visible=visible)
        sources_textboxes.append(t)
    return sources_textboxes
