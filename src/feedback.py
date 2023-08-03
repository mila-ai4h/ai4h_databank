from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pymongo
from buster.completers.base import Completion
from fastapi.encoders import jsonable_encoder
from pyparsing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class StandardForm:
    def to_json(self) -> Any:
        return jsonable_encoder(self)

    @classmethod
    def from_dict(cls, feedback_dict: dict) -> StandardForm:
        return cls(**feedback_dict)


@dataclass
class FeedbackForm(StandardForm):
    """Form on the original Buster app."""

    relevant_answer: str
    relevant_sources: str
    extra_info: str


@dataclass
class ComparisonForm(StandardForm):
    """Easily readable comparison result on the battle arena."""

    question: str
    model_left: str
    model_right: str
    vote: str


@dataclass
class Interaction:
    username: str
    user_completions: list[Completion]
    time: str
    feedback_form: Optional[StandardForm] = None

    def send(self, mongo_db: pymongo.database.Database, collection: str):
        feedback_json = self.to_json()
        logger.info(feedback_json)

        try:
            mongo_db[collection].insert_one(feedback_json)
            logger.info(f"response logged to mondogb {collection=}")
        except Exception as err:
            logger.exception(f"Something went wrong logging to mongodb {collection=}")
            raise err

    def flatten(self) -> dict:
        """Flatten feedback object into a dict for easier reading."""
        feedback_dict = self.to_json()

        # Flatten user completions, only keep the most recent interaction
        if len(feedback_dict["user_completions"]) > 0:
            completion_dict = feedback_dict["user_completions"][-1]
            # # TODO: add test for this...
            for k in completion_dict.keys():
                feedback_dict[f"completion_{k}"] = completion_dict[k]
        del feedback_dict["user_completions"]

        if self.feedback_form is not None:
            # Flatten feedback form
            for k in feedback_dict["feedback_form"].keys():
                feedback_dict[f"feedback_{k}"] = feedback_dict["feedback_form"][k]
            del feedback_dict["feedback_form"]

        # Flatten matched documents
        feedback_dict["matched_documents"] = self.user_completions[-1].matched_documents
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
            # Converts the matched_documents in the user_completions to json
            Completion: lambda completion: completion.to_json(columns_to_ignore=["embedding", "_id"]),
        }

        to_encode = {
            "username": self.username,
            "user_completions": self.user_completions,
            "time": self.time,
        }

        if self.feedback_form is not None:
            to_encode["feedback_form"] = self.feedback_form.to_json()

        return jsonable_encoder(to_encode, custom_encoder=custom_encoder)

    @classmethod
    def from_dict(cls, feedback_dict: dict, feedback_cls: StandardForm) -> Interaction:
        del feedback_dict["_id"]
        feedback_dict["feedback_form"] = feedback_cls.from_dict(feedback_dict["feedback_form"])

        feedback_dict["user_completions"] = [Completion.from_dict(r) for r in feedback_dict["user_completions"]]

        return cls(**feedback_dict)


def read_feedback(mongo_db: pymongo.database.Database, collection: str, filters: dict = None) -> pd.DataFrame:
    """Read feedback from mongodb.

    By default, return all feedback. If filters are provided, return only feedback that matches the filters.
    For example, to get just the feedback from a specific session, use filters={"username": <username>}.
    """
    try:
        feedback = mongo_db[collection].find(filters)
        feedback = [Interaction.from_dict(f).flatten() for f in feedback]
        feedback = pd.DataFrame(feedback)

        return feedback
    except Exception as err:
        logger.exception("Something went wrong reading from mongodb")
        return pd.DataFrame()
