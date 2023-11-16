import argparse
from typing import Type

import pandas as pd

from buster.documents_manager import DocumentsService
from buster.tokenizers import GPTTokenizer, Tokenizer
from buster.llm_utils import get_openai_embedding, BM25
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
    token_limit_per_chunk: int = 1000,
):
    # Make sure the chunks are not too big
    tokenizer = GPTTokenizer(buster_cfg.tokenizer_cfg["model_name"])
    dataframe["content"] = dataframe["content"].apply(lambda x: chunk_text(x, tokenizer, token_limit_per_chunk))
    dataframe = dataframe.explode("content", ignore_index=True)

    # Rename link to url
    dataframe.rename(columns={"link": "url"}, inplace=True)

    # Initialize BM25
    bm25 = BM25()
    bm25.fit(dataframe)
    bm25.dump_params("bm25_params.json")

    sparse_embedding_fn = bm25.get_sparse_embedding_fn()

    # Add the embeddings
    manager = DocumentsService(
        pinecone_api_key,
        pinecone_env,
        pinecone_index,
        pinecone_namespace,
        mongo_uri,
        mongo_db_data,
        required_columns=["content", "url", "title", "source", "country", "year"],
    )
    manager.batch_add(dataframe, embedding_fn=get_openai_embedding, sparse_embedding_fn=sparse_embedding_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload one or more CSV files containing chunks of data into the specified Pinecone namespace and Mongo database.\nUsage: python upload_data.py <pinecone_namespace> <mongo_db_name> <filepaths> --token_limit_per_chunk <token_limit_per_chunk>"
    )

    parser.add_argument("pinecone_namespace", type=str, help="Pinecone namespace to use.")
    parser.add_argument("mongo_db_data", type=str, help="Name of the mongo database to store the data.")
    parser.add_argument(
        "filepaths",
        type=str,
        nargs="+",
        help="Path to the csv containing the chunks. Will be loaded as a pandas dataframe.",
    )
    parser.add_argument("--token_limit_per_chunk", type=int, default=1000, help="Token limit per chunk. Default: 1000")

    args = parser.parse_args()

    # These are the new names
    pinecone_namespace = args.pinecone_namespace
    mongo_db_data = args.mongo_db_data
    token_limit_per_chunk = args.token_limit_per_chunk

    # Read data
    dataframes = [pd.read_csv(filepath, delimiter="\t") for filepath in args.filepaths]
    combined_dataframe = pd.concat(dataframes, ignore_index=True)

    upload_data(
        PINECONE_API_KEY,
        PINECONE_ENV,
        PINECONE_INDEX,
        pinecone_namespace,
        MONGO_URI,
        mongo_db_data,
        combined_dataframe,
        token_limit_per_chunk,
    )
