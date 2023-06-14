import pandas as pd

from src.cfg import retriever

def recall_at_k(df, k: int):
    """Computers average recall at K for a given df. Assumes only 1 positive document per retrieval."""
    # keep only positive results, -1 to be ignored
    df = df[df.results >= 0]
    return sum(df.results <= k) / len(df)

def get_top_k(matched_documents: pd.DataFrame, target_document) -> int:
    """For documents, find which k rank the target document is in. If not found, returns -1."""
    if any(contains_target := (matched_documents.content == target_document)):
        # Find the first index where the value is True
        top_k = contains_target.idxmax()
    else:
        top_k = -1
    return top_k

# args
questions_csv = "questions.csv"
max_k = 50

df = pd.read_csv(questions_csv)
top_k_results = []
for _, row in df.iterrows():
    question_text = row.question
    target_document = row.document

    matched_documents = retriever.get_topk_documents(row.question, source=None, top_k=max_k)

    top_k = get_top_k(matched_documents, target_document=row.document)

    top_k_results.append(top_k)

df["results"] = top_k_results

df.to_csv("results.csv", index=False)

top_k_counts = pd.Series(top_k_results).value_counts().sort_index()
print(top_k_results)
print(top_k_counts)

recall_at_k = {k: recall_at_k(df, k) for k in range(max_k)}
print(recall_at_k)