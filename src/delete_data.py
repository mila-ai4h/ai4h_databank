import argparse

from buster.documents_manager import DocumentsService
from src.cfg import MONGO_URI, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX


def delete_data(
    pinecone_api_key: str,
    pinecone_env: str,
    pinecone_index: str,
    pinecone_namespace: str,
    mongo_uri: str,
    mongo_db_data: str,
    source: str,
    drop_all: bool,
):
    manager = DocumentsService(
        pinecone_api_key,
        pinecone_env,
        pinecone_index,
        pinecone_namespace,
        mongo_uri,
        mongo_db_data,
    )
    if drop_all:
        manager.drop_db()
    else:
        manager.delete_source(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete data from the specified Pinecone namespace and Mongo database.\nUsage to delete a specific source: python delete_data.py <pinecone_namespace> <mongo_db_name> --source <source>\nUsage to drop the database: python delete_data.py <pinecone_namespace> <mongo_db_name> --all"
    )

    # Positional arguments
    parser.add_argument("pinecone_namespace", type=str, help="Pinecone namespace to use.")
    parser.add_argument("mongo_db_data", type=str, help="Name of the mongo database to store the data.")

    # Optional arguments
    parser.add_argument("--source", help="Specify a particular source to delete from the database.", type=str)
    parser.add_argument("--all", help="If set, it will drop the entire database.", action="store_true")

    args = parser.parse_args()

    # Check if both --source and --all are specified together
    if args.source and args.all:
        parser.error("The --source and --all options cannot be used together.")

    delete_data(
        PINECONE_API_KEY,
        PINECONE_ENV,
        PINECONE_INDEX,
        args.pinecone_namespace,
        MONGO_URI,
        args.mongo_db_data,
        args.source,
        args.all,
    )
