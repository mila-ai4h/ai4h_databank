from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pymongo
from buster.busterbot import Response
from buster.completers.base import Completion
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class FeedbackForm:
    good_bad: str
    extra_info: str
    relevant_answer: str
    relevant_length: str
    relevant_sources: str
    length_sources: str
    timeliness_sources: str
    version: int = 0

    @classmethod
    def from_dict(cls, feedback_dict: dict) -> FeedbackForm:
        # Backwards compatibility
        if "version" not in feedback_dict:
            feedback_dict["version"] = 0
        return cls(**feedback_dict)


@dataclass
class Feedback:
    session_id: str
    user_responses: list[Response]
    feedback_form: FeedbackForm
    time: str
    version: int = 1

    def to_json(self) -> Any:
        def encode_df(df: pd.DataFrame) -> str:
            if "embedding" in df.columns:
                df = df.drop(columns=["embedding"])
            return df.to_json(orient="index")

        custom_encoder = {
            # Converts the matched_documents in the user_responses to json
            pd.DataFrame: encode_df,
        }
        return jsonable_encoder(self, custom_encoder=custom_encoder)

    def send(self, mongo_db: pymongo.database.Database):
        feedback_json = self.to_json()
        logger.info(feedback_json)

        try:
            mongo_db["feedback"].insert_one(feedback_json)
            logger.info("response logged to mondogb")
        except Exception as err:
            logger.exception("Something went wrong logging to mongodb")

    @classmethod
    def from_dict(cls, feedback_dict: dict) -> Feedback:
        # Backwards compatibility
        if "version" not in feedback_dict:
            feedback_dict["version"] = 1

            feedback_dict["feedback_form"] = feedback_dict["feedback"]
            del feedback_dict["feedback"]

        del feedback_dict["_id"]
        feedback_dict["feedback_form"] = FeedbackForm.from_dict(feedback_dict["feedback_form"])

        # TODO move into buster Response and Completion as from_dict classmethod
        def response_from_dict(response_dict):
            response_dict["matched_documents"] = pd.DataFrame(response_dict["matched_documents"])
            response_dict["completion"] = Completion(**response_dict["completion"])
            return Response(**response_dict)

        feedback_dict["user_responses"] = [response_from_dict(r) for r in feedback_dict["user_responses"]]

        return cls(**feedback_dict)


def read_feedback(mongo_db: pymongo.database.Database, filters: dict = None) -> pd.DataFrame:
    """Read feedback from mongodb.

    By default, return all feedback. If filters are provided, return only feedback that matches the filters.
    For example, to get just the feedback from a specific session, use filters={"session_id": <session_id>}.
    """
    try:
        feedback = mongo_db["feedback"].find(filters)
        feedback = [Feedback.from_dict(f) for f in feedback]
        feedback = [flatten_feedback(f) for f in feedback]
        feedback = pd.DataFrame(feedback)
        feedback = feedback.drop_duplicates(subset=["session_id", "user_input"], keep="last")

        return feedback
    except Exception as err:
        logger.exception("Something went wrong reading from mongodb")
        return []


def flatten_feedback(feedback: Feedback) -> dict:
    """Flatten feedback object into a dict for easier reading."""
    feedback_dict = feedback.to_json()

    # Flatten user responses, only keep the most recent interaction
    if len(feedback_dict["user_responses"]) > 0:
        feedback_dict.update(feedback_dict["user_responses"][-1])

        for k in feedback_dict["completion"].keys():
            feedback_dict[f"completion_{k}"] = feedback_dict["completion"][k]
        del feedback_dict["completion"]
    del feedback_dict["user_responses"]

    # Flatten feedback form
    for k in feedback_dict["feedback_form"].keys():
        feedback_dict[f"feedback_{k}"] = feedback_dict["feedback_form"][k]
    del feedback_dict["feedback_form"]

    # Flatten matched documents
    feedback_dict["matched_documents"] = feedback.user_responses[-1].matched_documents.T
    feedback_dict["matched_documents"].reset_index(inplace=True)
    feedback_dict["matched_documents"].drop(columns=["index"], inplace=True)
    feedback_dict["matched_documents"] = feedback_dict["matched_documents"].T
    if len(feedback_dict["matched_documents"]) > 0:
        for k in feedback_dict["matched_documents"].keys():
            feedback_dict[f"matched_documents_{k}"] = feedback_dict["matched_documents"][k].values
        del feedback_dict["matched_documents"]

    return feedback_dict
