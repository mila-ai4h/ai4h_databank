import argparse
from typing import Type

import pandas as pd

from buster.documents_manager import DocumentsService
from buster.tokenizers import GPTTokenizer, Tokenizer
from src.cfg import (
    MONGO_URI,
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX,
    buster_cfg,
)


def chunk_text(text: str, tokenizer: Type[Tokenizer], token_limit: int) -> list[str]:
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), token_limit):
        chunks.append(tokenizer.decode(tokens[i : i + token_limit]))

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
    tokenizer = GPTTokenizer(buster_cfg.tokenizer_cfg["model_name"])
    token_limit_per_chunk = 1000
    dataframe["content"] = dataframe["content"].apply(lambda x: chunk_text(x, tokenizer, token_limit_per_chunk))
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
