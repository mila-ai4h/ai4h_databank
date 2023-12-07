import argparse
import datetime

import pandas as pd
from matplotlib import pyplot as plt

import src.cfg as cfg
from src.app_utils import get_logging_db_name, init_db
from src.feedback import FeedbackForm, read_collection


def get_collection_and_feedback_cls(collection_name: str):
    """Gets the correct collection and feedback class based on collection_name.

    Args:
        collection_name: The name of the collection. Should be one of ['interaction', 'feedback', 'flagged'].

    Returns:
        The collection and feedback class associated with the collection_name.

    Raises:
        ValueError: If collection_name is not one of ['interaction', 'feedback', 'flagged'].
    """
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

    return collection, feedback_cls


def dump_collection(collection_name: str, time: str = None) -> None:
    """Dumps collection data into a csv file.

    Args:
        collection_name: The name of the collection to dump. Should be one of ['interaction', 'feedback', 'flagged'].
        time: An optional parameter indicating the minimum time for the dumped data.

    Returns:
        None
    """
    mongo_db = init_db(mongo_uri=cfg.MONGO_URI, db_name=get_logging_db_name("prod"))

    filters = {"time": {"$gt": time}} if time is not None else None

    collection, feedback_cls = get_collection_and_feedback_cls(collection_name=collection_name)
    df = read_collection(mongo_db, collection=collection, feedback_cls=feedback_cls, filters=filters)

    current_date = datetime.date.today().strftime("%Y-%m-%d")
    df_name = f"{collection_name}_{current_date}.csv"
    df.to_csv(df_name, index=False)
    print(f"Succesfully dumped {collection_name=} to {df_name}")


def get_daily_counts(df: pd.DataFrame) -> pd.Series:
    """Gets the number of occurrences on a given date in a dataframe.

    Args:
        df: The dataframe to calculate the daily counts from.

    Returns:
        A series containing the number of occurrences on each date.
    """
    df["time"] = pd.to_datetime(df["time"])
    return df.groupby(df["time"].dt.date).size()


def plot_daily_usage(collections: list[str] = ["interaction", "feedback"]):
    """Plots the daily usage of collections.

    Args:
        collections: A list of collection names to plot the daily usage for. Default is ['interaction', 'feedback'].

    Returns:
        None
    """
    mongo_db = init_db(mongo_uri=cfg.MONGO_URI, db_name=get_logging_db_name("prod"))
    for collection_name in collections:
        collection, feedback_cls = get_collection_and_feedback_cls(collection_name=collection_name)
        df = read_collection(mongo_db, collection=collection, feedback_cls=feedback_cls)
        daily_counts = get_daily_counts(df)
        daily_counts.plot()

    plt.title("Daily Interactions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Interactions")
    plt.grid(True)
    plt.legend(["User Interactions", "Feedback Forms"])
    plt.xticks(rotation=45)
    plt.savefig("daily_usage.png", bbox_inches="tight")
    plt.show()


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
