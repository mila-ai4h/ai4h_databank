import os

import pandas as pd

from src.db_utils import init_db
from src.feedback import Feedback, FeedbackForm, read_feedback

"""
Fetch the feedback from the round of human evaluation done on gpt-3.5-turbo on 2023-07-04.

Some of the databank members used the dev server so we also fetch the evaluations from there.
"""


if __name__ == "__main__":
    # shared envs.
    username = os.getenv("AI4H_MONGODB_USERNAME")
    password = os.getenv("AI4H_MONGODB_PASSWORD")
    cluster = os.getenv("AI4H_MONGODB_CLUSTER")

    # prod envs
    db_name = "ai4h-databank-prod"
    mongo_db_prod = init_db(username, password, cluster, db_name)
    collection_prod = "feedback-04072023"
    filters_prod = None

    # dev envs
    db_name = "ai4h-databank-dev"
    mongo_db_dev = init_db(username, password, cluster, db_name)
    collection_dev = "feedback-dev"
    # they ended up using the dev db as well for evaluations, so fetch the feedback there
    filters_dev = {"username": {"$ne": "databank-allison"}}

    df_1 = read_feedback(mongo_db_dev, collection=collection_dev, filters=filters_dev)
    df_2 = read_feedback(mongo_db_prod, collection=collection_prod, filters=filters_prod)

    # Join them together, save to csv.
    df = pd.concat([df_1, df_2])
    print("Number of documents retrieved: ", len(df))
    df.to_csv("feedback_04072023_dump.csv", index=False)

    # print some overall stats of the feedback split by users
    relevant_answers = df.groupby("username").apply(lambda x: sum(x.feedback_relevant_answer == "👍") / len(x))
    relevant_sources = df.groupby("username").apply(lambda x: sum(x.feedback_relevant_sources == "👍") / len(x))
    num_entries = df.groupby("username").apply(lambda x: len(x))

    results = pd.concat([relevant_answers, relevant_sources, num_entries], axis=1)
    results.columns = ["relevant_answers", "relevant_sources", "total_entries"]
    print(results)
