import pandas as pd
from tqdm import tqdm

from buster.retriever import Retriever
from src.cfg import retriever

# get access to pandas tqdm progress bar
tqdm.pandas()


def recall_at_k(df, k: int):
    """Computes average recall at K over all samples in given df. Assumes only 1 positive document per retrieval."""
    return sum(df.doc_rank <= k) / len(df)


def get_document_rank(x, top_k: int, retriever: Retriever) -> int | float:
    """Computes the rank at which a given document was retrieved.

    x is a pandas row, which should contain a question, x.question and document, x.document to be retrieved.

    If the document is not found in the retrieved top_k documents, it is considered inf.
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
    # args
    questions_csv = "questions.csv"  # generated questions using the generate_questions.py script
    top_k = 50  # max number of documents to retrieve

    # load the questions
    df = pd.read_csv(questions_csv)
    print(f"Computing Recall@K for {questions_csv}. Found {len(questions_csv)} questions.")

    # Computes the retrieved document's rank.
    # progress_apply replaces apply with a pandas tqdm wrapper
    df["doc_rank"] = df.progress_apply(get_document_rank, args=(top_k, retriever), axis=1)

    df.to_csv("results.csv", index=False)

    print(df.doc_rank)

    # Prints the frequency of a given count
    rank_counts = df.doc_rank.value_counts().sort_index()
    print(rank_counts)

    recall_at_k_results = {k: recall_at_k(df, k) for k in range(top_k)}
    print(f"{recall_at_k_results}")


if __name__ == "__main__":
    main()
