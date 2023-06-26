from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pymongo
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

    def to_json(self) -> Any:
        return jsonable_encoder(self)

    @classmethod
    def from_dict(cls, feedback_dict: dict) -> FeedbackForm:
        # Backwards compatibility
        if "version" not in feedback_dict:
            feedback_dict["version"] = 0
        return cls(**feedback_dict)


@dataclass
class Feedback:
    session_id: str
    user_responses: list[Completion]
    feedback_form: FeedbackForm
    time: str
    version: int = 1

    def send(self, mongo_db: pymongo.database.Database):
        feedback_json = self.to_json()
        logger.info(feedback_json)

        try:
            mongo_db["feedback"].insert_one(feedback_json)
            logger.info("response logged to mondogb")
        except Exception as err:
            logger.exception("Something went wrong logging to mongodb")

    def flatten(self) -> dict:
        """Flatten feedback object into a dict for easier reading."""
        feedback_dict = self.to_json()

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
        feedback_dict["matched_documents"] = self.user_responses[-1].matched_documents
        feedback_dict["matched_documents"].reset_index(inplace=True)
        feedback_dict["matched_documents"].drop(columns=["index"], inplace=True)
        feedback_dict["matched_documents"] = feedback_dict["matched_documents"].T
        if len(feedback_dict["matched_documents"]) > 0:
            for k in feedback_dict["matched_documents"].keys():
                feedback_dict[f"matched_documents_{k}"] = feedback_dict["matched_documents"][k].values
        del feedback_dict["matched_documents"]

        return feedback_dict

    def to_json(self) -> Any:

        custom_encoder = {
            # Converts the matched_documents in the user_responses to json
            Completion: lambda completion: completion.to_json(columns_to_ignore=["embedding", "_id"]),
        }

        to_encode = {
            "session_id": self.session_id,
            "user_responses": self.user_responses,
            "feedback_form": self.feedback_form.to_json(),
            "time": self.time,
            "version": self.version,
        }
        return jsonable_encoder(to_encode, custom_encoder=custom_encoder)

    @classmethod
    def from_dict(cls, feedback_dict: dict) -> Feedback:
        # Backwards compatibility
        if "version" not in feedback_dict:
            feedback_dict["version"] = 1

            feedback_dict["feedback_form"] = feedback_dict["feedback"]
            del feedback_dict["feedback"]

        del feedback_dict["_id"]
        feedback_dict["feedback_form"] = FeedbackForm.from_dict(feedback_dict["feedback_form"])

        feedback_dict["user_responses"] = [Completion.from_dict(r) for r in feedback_dict["user_responses"]]

        return cls(**feedback_dict)


def read_feedback(mongo_db: pymongo.database.Database, filters: dict = None) -> pd.DataFrame:
    """Read feedback from mongodb.

    By default, return all feedback. If filters are provided, return only feedback that matches the filters.
    For example, to get just the feedback from a specific session, use filters={"session_id": <session_id>}.
    """
    try:
        feedback = mongo_db["feedback"].find(filters)
        feedback = [Feedback.from_dict(f).flatten() for f in feedback]
        feedback = pd.DataFrame(feedback)
        feedback = feedback.drop_duplicates(subset=["session_id", "user_input"], keep="last")

        return feedback
    except Exception as err:
        logger.exception("Something went wrong reading from mongodb")
        return pd.DataFrame()
