import argparse
import os
from typing import List, Type

import pandas as pd
from huggingface_hub import HfApi

from buster.documents_manager import DeepLakeDocumentsManager, DocumentsService
from buster.llm_utils.embeddings import get_openai_embedding_constructor
from buster.tokenizers import GPTTokenizer, Tokenizer
from src.cfg import buster_cfg

# the embedding function that will get used to embed documents in the app
embedding_fn = get_openai_embedding_constructor(model="text-embedding-ada-002", client_kwargs={"max_retries": 10})


def upload_to_hf(path_or_fileobj, repo_id: str = None, path_in_repo: str = None):
    """Uploads a file or object to the Huggingface dataset.

    Args:
        path_or_fileobj: The path to the file or file-like object to upload.
    """
    # Get the details specified in cfg.py
    from src.cfg import HF_DATASET_REPO_ID, HF_TOKEN, HF_VECTOR_STORE_PATH

    if repo_id is None:
        repo_id = HF_DATASET_REPO_ID

    if path_in_repo is None:
        path_in_repo = HF_VECTOR_STORE_PATH

    print(f"Uploading {path_or_fileobj} to Huggingface dataset {repo_id} as {path_in_repo}")
    api = HfApi()
    api.create_repo(repo_id=HF_DATASET_REPO_ID, private=True, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=path_or_fileobj,
        path_in_repo=path_in_repo,
        token=HF_TOKEN,
    )


def get_user_confirmation() -> bool:
    """Asks the user for confirmation to proceed. Returns True if 'y' is entered, otherwise False."""
    user_input = input("Do you want to continue? (y/[n]): ").strip().lower()
    return user_input == "y"


def get_files_to_upload(filepaths: List[str] = None, directory: str = None) -> List[str]:
    """Returns a list of files to upload.

    Args:
        filepaths: A list of file paths to upload.
        directory: The directory where files can be found.

    Returns:
        A list of file paths to upload.
    """
    files_to_upload = []
    if filepaths is not None:
        files_to_upload.extend(filepaths)

    if directory is not None:
        files_to_upload.extend(get_files_with_extensions(directory, extensions=[".txt", ".csv"]))

    return files_to_upload


def get_files_with_extensions(directory: str, extensions: List[str]) -> List[str]:
    """Recursively searches for files with specific extensions in a directory.

    Args:
        directory: The directory path to search in.
        extensions: A list of file extensions to look for.

    Returns:
        A list of file paths matching the specified extensions.
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                matching_files.append(os.path.join(root, file))
    return matching_files


def chunk_text(text: str, tokenizer: Type[Tokenizer], token_limit: int) -> List[str]:
    """Chunks a given text into tokens with a specified token limit.

    Args:
        text: The text to chunk.
        tokenizer: The tokenizer to use.
        token_limit: The maximum number of tokens per chunk.

    Returns:
        A list of text chunks.
    """
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
    """Uploads data to the document manager.

    Args:
        manager: The document manager.
        dataframe: The data to upload.
        token_limit_per_chunk: The token limit per chunk.
    """
    # Make sure the chunks are not too big
    tokenizer = GPTTokenizer(buster_cfg.tokenizer_cfg["model_name"])
    dataframe["content"] = dataframe["content"].apply(lambda x: chunk_text(x, tokenizer, token_limit_per_chunk))
    dataframe = dataframe.explode("content", ignore_index=True)

    # Rename link to url
    dataframe.rename(columns={"link": "url"}, inplace=True)

    manager.batch_add(
        df=dataframe,
        batch_size=2900,
        embedding_fn=embedding_fn,
    )


def main():
    """Uploads one or more CSV files containing chunks of data using the specified document manager
    (supports DeepLake and Pinecone/MongoDB)."""
    parser = argparse.ArgumentParser(
        description="Upload one or more CSV files containing chunks of data using the specified document manager (supports DeepLake and Pinecone/MongoDB). \nUsage: python upload_data.py --args ..."
    )

    parser.add_argument(
        "--filepaths",
        type=str,
        nargs="+",
        help="Path to the CSV containing the chunks. Will be loaded as a pandas dataframe. Specify files one at a time.",
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
        help="Which manager to use; Pinecone+MongoDB ('service')  or DeepLake ('deeplake')",
        default="deeplake",
    )

    args = parser.parse_args()

    token_limit_per_chunk = args.token_limit_per_chunk

    files_to_upload = get_files_to_upload(filepaths=args.filepaths, directory=args.directory)

    # Read data
    dataframes = [pd.read_csv(f, delimiter="\t") for f in files_to_upload]
    combined_dataframe = pd.concat(dataframes, ignore_index=True)

    print(f"Files to be uploaded: {files_to_upload}")
    print(
        f"Total number of documents that will be computed: {len(combined_dataframe)}. Note that this number is approximate and might change based on 'token_limit_per_chunk'"
    )

    confirmation = get_user_confirmation()

    if not confirmation:
        print("Aborted by user.")
        return

    if args.document_manager == "service":
        from src.cfg import (
            DEEPLAKE_VECTOR_STORE_PATH,
            MONGO_DATABASE_DATA,
            MONGO_URI,
            PINECONE_API_KEY,
            PINECONE_ENV,
            PINECONE_INDEX,
            PINECONE_NAMESPACE,
        )

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
        from src.cfg import DEEPLAKE_VECTOR_STORE_PATH

        if os.path.exists(DEEPLAKE_VECTOR_STORE_PATH):
            print(
                f"""
                WARNING: found existing DeepLake vector store at {DEEPLAKE_VECTOR_STORE_PATH}.
                This will duplicate embeddings if they've already been passed.
                Consider specifying a new documents version or deleting the previous version.
                You can safely ignore this message if adding additional embeddings.

                """
            )

            confirmation = get_user_confirmation()

            if not confirmation:
                return

        print(f"DeepLake dataset will be saved to {DEEPLAKE_VECTOR_STORE_PATH}.")
        document_manager = DeepLakeDocumentsManager(
            vector_store_path=DEEPLAKE_VECTOR_STORE_PATH,
            required_columns=["content", "url", "title", "source", "country", "year"],
        )

        upload_data(
            manager=document_manager,
            dataframe=combined_dataframe,
            token_limit_per_chunk=token_limit_per_chunk,
        )

        zip_fname = document_manager.to_zip()
        print(f"Successfully zipped the DeepLake dataset to {zip_fname}")

        # Upload to Huggingface data space
        upload_to_hf(path_or_fileobj=zip_fname)

    else:
        raise ValueError(f"document_manager must be one of ['deeplake', 'service']. Provided: {args.document_manager}")


if __name__ == "__main__":
    main()
