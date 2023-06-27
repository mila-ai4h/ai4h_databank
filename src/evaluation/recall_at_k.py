import json

import pandas as pd
from buster.retriever import Retriever
from tqdm import tqdm

from src.cfg import retriever

# get access to pandas tqdm progress bar
tqdm.pandas()


def recall_at_k(df, k: int, column: str) -> float:
    """Computes the average recall at K over all samples in the given DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the data.
        k (int): The cutoff rank value for computing recall.
        column (str): The name of the column in the DataFrame that represents the document ranks.

    Returns:
        float: The average recall at K over all samples.

    Assumes:
        - Each retrieval has only one positive document.
    """
    return sum(df[column] <= k) / len(df)


def get_document_rank(x, top_k: int, retriever: Retriever) -> int | float:
    """
    Computes the rank at which a given document was retrieved.

    Args:
        x (pandas.Series): A pandas row containing a question and document to be retrieved.
        top_k (int): The total number of matched documents to retrieve.
        retriever (Retriever): The Retriever object used to retrieve documents.

    Returns:
        int | float: The rank at which the document was retrieved. If the document is not found
                     in the retrieved top_k documents, it is considered as infinity (inf).
    """



    # retrieve the top_k matched documents
    matched_documents = retriever.get_topk_documents(x.question, source=None, top_k=top_k)

    # find rank of known document
    if any(contains_target := (matched_documents.content == x.document)):
        # Find the first index where the value is True
        rank = contains_target.idxmax()
    else:
        rank = float("inf")
    return rank


def main():
    """Compute Recall@K and rank statistics for a set of questions.

    This function reads questions from a CSV file generated using the 'generate_questions.py' script.
    It retrieves a maximum of 'top_k' documents for each question, computes the rank of the retrieved document,
    saves the results to a CSV file, and prints the rank frequency and Recall@K results.
    """

    # args
    questions_csv = "questions.csv"  # generated questions using the generate_questions.py script
    top_k = 50  # max number of documents to retrieve

    # load the questions
    df = pd.read_csv(questions_csv)
    print(f"Computing Recall@K for {questions_csv}. Found {len(questions_csv)} questions.")

    # Computes the retrieved document's rank.
    # progress_apply replaces apply with a pandas tqdm wrapper
    doc_rank_column = "doc_rank"
    df[doc_rank_column] = df.progress_apply(get_document_rank, args=(top_k, retriever), axis=1)

    # Display the results
    print(df)

    # save the result to .csv
    df.to_csv("results_tmp.csv", index=False)

    # Compute and display the frequency of a rank
    rank_counts = df[doc_rank_column].value_counts().sort_index()
    print(rank_counts)

    # compute the recall at k results
    recall_at_k_results = {k: recall_at_k(df, k, column=doc_rank_column) for k in range(top_k)}
    print(f"{recall_at_k_results}")

    # Dump the results to a .json file
    with open("recall_at_k_results.json", "w") as file:
        json.dump(recall_at_k_results, file)



if __name__ == "__main__":
    main()
