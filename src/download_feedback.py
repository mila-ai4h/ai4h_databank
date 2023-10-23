import argparse
import datetime

import pandas as pd

import src.cfg as cfg
from src.app_utils import get_logging_db_name, init_db
from src.feedback import FeedbackForm, read_collection


def download_feedback(time: str = None) -> None:
    mongo_db = init_db(mongo_uri=cfg.MONGO_URI, db_name=get_logging_db_name("prod"))

    filters = {"time": {"$gt": time}} if time is not None else None

    df = read_collection(mongo_db, cfg.MONGO_COLLECTION_FEEDBACK, feedback_cls=FeedbackForm, filters=filters)

    current_date = datetime.date.today().strftime("%Y-%m-%d")
    df_name = f"feedback_{current_date}.csv"
    df.to_csv(df_name, index=False)


def download_interactions(filters=None) -> pd.DataFrame:
    mongo_db = init_db(mongo_uri=cfg.MONGO_URI, db_name=get_logging_db_name("prod"))

    df = read_collection(
        mongo_db,
        cfg.MONGO_COLLECTION_INTERACTION,
        feedback_cls=None,
        filters=filters,
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Collect feedback from MongoDB prod. The output is a csv file named "feedback_YYYY-MM-DD.csv" where YYYY-MM-DD is the current date.'
    )
    parser.add_argument(
        "--time",
        type=str,
        help="If specified, will filter the collection for feedback as or more recent than this time. Expected format is YYYY-MM-DD.",
    )

    args = parser.parse_args()

    download_feedback(args.time)
