import argparse
import os
from typing import Type

import pandas as pd

from buster.documents_manager import DocumentsService, DeepLakeDocumentsManager
from buster.tokenizers import GPTTokenizer, Tokenizer
from src.cfg import (
    MONGO_URI,
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX,
    DEEPLAKE_VECTOR_STORE_PATH,
    PINECONE_NAMESPACE,
    MONGO_DATABASE_DATA,
    buster_cfg,
)


def get_user_confirmation() -> bool:
    """
    Asks the user for a confirmation to proceed. Only continues if 'y' is entered.

    Returns:
    bool: True if the user enters 'y', otherwise False.
    """
    user_input = input("Do you want to continue? (y/[n]): ").strip().lower()
    return user_input == "y"


def get_files_to_upload(filepaths: list[str] = None, directory: str = None):
    # Get all files to upload
    files_to_upload = []
    if filepaths is not None:
        files_to_upload.extend(filepaths)

    if directory is not None:
        files_to_upload.extend(get_files_with_extensions(directory, extensions=[".txt", ".csv"]))

    return files_to_upload


def get_files_with_extensions(directory: str, extensions: list[str]) -> list[str]:
    """
    Recursively searches for files with specific extensions in a directory.

    Args:
    directory (str): The directory path to search in.
    extensions (List[str]): A list of file extensions to look for.

    Returns:
    List[str]: A list of file paths matching the specified extensions.
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                matching_files.append(os.path.join(root, file))
    return matching_files


# Example usage
# directory_to_search = "path/to/your/directory"
# extensions_to_search = [".txt", ".csv"]
# files = get_files_with_extensions(directory_to_search, extensions_to_search


def chunk_text(text: str, tokenizer: Type[Tokenizer], token_limit: int) -> list[str]:
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), token_limit):
        chunks.append(tokenizer.decode(tokens[i : i + token_limit]))

    return chunks


def upload_data(
    manager,
    dataframe: pd.DataFrame,
    token_limit_per_chunk: int = 1000,
):
    # Make sure the chunks are not too big
    tokenizer = GPTTokenizer(buster_cfg.tokenizer_cfg["model_name"])
    dataframe["content"] = dataframe["content"].apply(lambda x: chunk_text(x, tokenizer, token_limit_per_chunk))
    dataframe = dataframe.explode("content", ignore_index=True)

    # Rename link to url
    dataframe.rename(columns={"link": "url"}, inplace=True)

    manager.batch_add(dataframe)


def main():
    parser = argparse.ArgumentParser(
        description="Upload one or more CSV files containing chunks of data into the specified Pinecone namespace and Mongo database.\nUsage: python upload_data.py <pinecone_namespace> <mongo_db_name> <filepaths> --token_limit_per_chunk <token_limit_per_chunk>"
    )

    parser.add_argument(
        "--filepaths",
        type=str,
        nargs="+",
        help="Path to the csv containing the chunks. Will be loaded as a pandas dataframe. Specify files one at a time.",
        default=None,
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to a directory containing all files to upload. Will look for .csv and .txt files recursively.",
        default=None,
    )
    parser.add_argument("--token_limit_per_chunk", type=int, default=1000, help="Token limit per chunk. Default: 1000")
    parser.add_argument(
        "--document-manager",
        type=str,
        help="Which manager to use; pinecone+mongo ('service')  or deeplake ('deeplake')",
        default="deeplake",
    )

    args = parser.parse_args()

    token_limit_per_chunk = args.token_limit_per_chunk

    files_to_upload = get_files_to_upload(filepaths=args.filepaths, directory=args.directory)

    print(f"Files to be uploaded: {files_to_upload}")

    confirmation = get_user_confirmation()

    if not confirmation:
        print("Aborted by user.")
        return

    # Read data
    dataframes = [pd.read_csv(f, delimiter="\t") for f in files_to_upload]
    combined_dataframe = pd.concat(dataframes, ignore_index=True)

    if args.document_manager == "service":
        document_manager = DocumentsService(
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_env=PINECONE_ENV,
            pinecone_index=PINECONE_INDEX,
            pinecone_namespace=PINECONE_NAMESPACE,
            mongo_uri=MONGO_URI,
            mongo_db_data=MONGO_DATABASE_DATA,
            required_columns=["content", "url", "title", "source", "country", "year"],
        )

        upload_data(
            manager=document_manager,
            dataframe=combined_dataframe,
            token_limit_per_chunk=token_limit_per_chunk,
        )
    elif args.document_manager == "deeplake":
        if os.path.exists(DEEPLAKE_VECTOR_STORE_PATH):
            print(
                f"""
                WARNING: found existing deeplake vector store at {DEEPLAKE_VECTOR_STORE_PATH}.
                This will duplicate embeddings if they've already been passed.
                Consider specifying a new documents version or deleting the previous version.
                You can safely ignore this message if adding additional embeddings.

                """
            )

            confirmation = get_user_confirmation()

            if not confirmation:
                return

        document_manager = DeepLakeDocumentsManager(
            vector_store_path=DEEPLAKE_VECTOR_STORE_PATH,
            required_columns=["content", "url", "title", "source", "country", "year"],
        )

        upload_data(
            manager=document_manager,
            dataframe=combined_dataframe,
            token_limit_per_chunk=token_limit_per_chunk,
        )

        # Upload to huggingface data space?
    else:
        raise ValueError(f"document_manager must be one of ['deeplake', 'service']. Provided: {args.document_manager}")


if __name__ == "__main__":
    main()
