import argparse
import datetime

from src.app_utils import get_logging_db_name, init_db
from src.cfg import MONGO_COLLECTION_FEEDBACK, MONGO_URI
from src.feedback import FeedbackForm, read_collection


def collect_feedback(time: str = None):
    mongo_db = init_db(mongo_uri=MONGO_URI, db_name=get_logging_db_name("prod"))

    df = read_collection(
        mongo_db, MONGO_COLLECTION_FEEDBACK, feedback_cls=FeedbackForm, filters={"time": {"$gt": time}}
    )

    current_date = datetime.date.today().strftime("%Y-%m-%d")
    df_name = f"feedback_{current_date}.csv"
    df.to_csv(df_name, index=False)


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

    collect_feedback(args.time)
