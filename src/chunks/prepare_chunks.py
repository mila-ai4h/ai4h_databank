import pandas as pd
from buster.documents_manager import DeepLakeDocumentsManager
from langchain.text_splitter import RecursiveCharacterTextSplitter


def word_length(x: str):
    return len(x.split())


def chunkify_recursive(x: str, **kwargs) -> list[str]:
    """Split the input text into chunks using the RecursiveCharacterTextSplitter.

    :param x: Input text to be split.
    :param kwargs: Additional arguments for RecursiveCharacterTextSplitter.
    :return: List of split chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(**kwargs)

    chunks = text_splitter.create_documents([x])
    chunks_list = [c.page_content for c in chunks]
    return chunks_list


def aggregate_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rows in the df such that 'content' for a given url is all contained in one row.

    For all other rows, returns the first element of their unique set.

    :param df: Input DataFrame containing chunks.
    :return: Aggregated DataFrame.
    """

    df = df.sort_values(by=["url", "title"])

    # for every col, simply take the first element of the unique set
    agg_funcs = {col: lambda x: x.unique()[0] for col in df.columns}

    # ... except for the content function, in which case we join all previous chunks which were ordered
    agg_funcs["content"] = lambda x: " ".join(x.to_list())

    agg_df = df.groupby("url").agg(agg_funcs)
    agg_df = agg_df.reset_index(drop=True)
    return agg_df


def chunkify_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the content in the DataFrame into chunks and explode to individual rows.

    :param df: Input DataFrame.
    :return: DataFrame with exploded chunks.
    """
    kwargs = {
        "chunk_size": 750,
        "separators": ["\n", ".", " "],
        "chunk_overlap": 100,
        "length_function": word_length,
    }

    # Split content into chunks using the selected strategy
    df["content"] = df["content"].apply(chunkify_recursive, **kwargs)

    # Explode "content" column to individual rows
    df = df.explode("content")
    kwargs = {
        "chunk_size": 750,
        "separators": ["\n", ".", " "],
        "chunk_overlap": 100,
        "length_function": word_length,
    }

    # chunkify using the selected langchain strategy
    df["content"] = df["content"].apply(chunkify_recursive, **kwargs)

    # in previous operation, "content" is now a list of chunks. Exploding will make each item its own row
    df = df.explode("content")

    return df


def main():
    # sample usage
    df = pd.read_csv("path/to/csv")
    df = df.dropna(subset="content")
    df = df.rename({"link": "url"}, axis=1)
    df = aggregate_chunks(df)
    df = chunkify_df(df)

    df.to_csv("new_chunks.csv")
