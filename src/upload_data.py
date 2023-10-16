import argparse
import os

import pandas as pd

from buster.documents_manager import DocumentsService
from src.app_utils import make_uri
from src.cfg import MONGO_URI, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX


def split_text(text: str, max_words: int = 500) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def upload_data(
    pinecone_api_key: str,
    pinecone_env: str,
    pinecone_index: str,
    pinecone_namespace: str,
    mongo_uri: str,
    mongo_db_data: str,
    dataframe: pd.DataFrame,
):
    # Make sure the chunks are not too big
    dataframe["content"] = dataframe["content"].apply(split_text)
    dataframe = dataframe.explode("content", ignore_index=True)

    # Rename link to url
    dataframe.rename(columns={"link": "url"}, inplace=True)

    manager = DocumentsService(
        pinecone_api_key,
        pinecone_env,
        pinecone_index,
        pinecone_namespace,
        mongo_uri,
        mongo_db_data,
        required_columns=["content", "url", "title", "source", "country", "year"],
    )
    manager.batch_add(dataframe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a CSV file containing chunks of data into the specified Pinecone namespace and Mongo database.\nUsage: python upload_data.py <pinecone_namespace> <mongo_db_name> <filepath>"
    )

    parser.add_argument("pinecone_namespace", type=str, help="Pinecone namespace to use.")
    parser.add_argument("mongo_db_data", type=str, help="Name of the mongo database to store the data.")
    parser.add_argument(
        "filepath", type=str, help="Path to the csv containing the chunks. Will be loaded as a pandas dataframe."
    )

    args = parser.parse_args()

    # These are the new names
    pinecone_namespace = args.pinecone_namespace
    mongo_db_data = args.mongo_db_data

    # Read data
    df = pd.read_csv(args.filepath, delimiter="\t")

    upload_data(PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX, pinecone_namespace, MONGO_URI, mongo_db_data, df)
