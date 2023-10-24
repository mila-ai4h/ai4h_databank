import argparse
import datetime

import pandas as pd

import src.cfg as cfg
from src.app_utils import get_logging_db_name, init_db
from src.feedback import FeedbackForm, read_collection


def dump_collection(collection_name: str, time: str = None) -> None:
    mongo_db = init_db(mongo_uri=cfg.MONGO_URI, db_name=get_logging_db_name("prod"))

    filters = {"time": {"$gt": time}} if time is not None else None

    match collection_name:
        case "interaction":
            collection = cfg.MONGO_COLLECTION_INTERACTION
            feedback_cls = None
        case "feedback":
            collection = cfg.MONGO_COLLECTION_FEEDBACK
            feedback_cls = FeedbackForm
        case "flagged":
            collection = cfg.MONGO_COLLECTION_FLAGGED
            feedback_cls = None
        case _:
            raise ValueError(f"collection_name should be in ['interaction', 'feedback', 'flagged']")

    df = read_collection(mongo_db, collection=collection, feedback_cls=feedback_cls, filters=filters)

    current_date = datetime.date.today().strftime("%Y-%m-%d")
    df_name = f"{collection_name}_{current_date}.csv"
    df.to_csv(df_name, index=False)
    print(f"Succesfully dumped {collection_name=} to {df_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Collect feedback from MongoDB prod. The output is a csv file named "feedback_YYYY-MM-DD.csv" where YYYY-MM-DD is the current date.'
    )
    parser.add_argument(
        "collection_name",
        type=str,
        choices=["interaction", "feedback", "flagged"],
        help="The name of the collection to dump. Choose from ['interaction', 'feedback', 'flagged'].",
    )
    parser.add_argument(
        "--time",
        type=str,
        help="If specified, will filter the collection for feedback as or more recent than this time. Expected format is YYYY-MM-DD.",
    )

    args = parser.parse_args()

    dump_collection(args.collection_name, args.time)
