from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Type

import pandas as pd
import pymongo
from fastapi.encoders import jsonable_encoder
from pyparsing import Optional

from buster.completers.base import Completion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class StandardForm:
    def to_json(self) -> Any:
        return jsonable_encoder(self)

    @classmethod
    def from_dict(cls, interaction_dict: dict) -> StandardForm:
        return cls(**interaction_dict)


@dataclass
class FeedbackForm(StandardForm):
    """Form on the original Buster app."""

    # Overall experience
    overall_experience: str

    # Answer Quality
    clear_answer: str
    accurate_answer: str

    # Source Relevance
    relevant_sources: str
    relevant_sources_order: str
    relevant_sources_selection: list

    # Additional Feedback
    extra_info: str


@dataclass
class ComparisonForm(StandardForm):
    """Easily readable comparison result on the battle arena."""

    question: str
    model_left: str
    model_right: str
    vote: str
    extra_info: str


@dataclass
class Interaction:
    username: str
    user_completions: list[Completion]
    time: str
    form: Optional[StandardForm] = None

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
        """Flattens the Interaction object into a dict for easier reading."""
        interaction_dict = self.to_json()

        # Flatten user completions, only keep the most recent interaction
        if len(interaction_dict["user_completions"]) > 0:
            completion_dict = interaction_dict["user_completions"][-1]
            # # TODO: add test for this...
            for k in completion_dict.keys():
                interaction_dict[f"completion_{k}"] = completion_dict[k]
        del interaction_dict["user_completions"]

        if self.form is not None:
            # Flatten feedback form
            for k in interaction_dict["form"].keys():
                interaction_dict[f"form_{k}"] = interaction_dict["form"][k]
            del interaction_dict["form"]

        # Flatten matched documents
        interaction_dict["matched_documents"] = self.user_completions[-1].matched_documents
        interaction_dict["matched_documents"].reset_index(inplace=True)
        interaction_dict["matched_documents"].drop(columns=["index"], inplace=True)
        interaction_dict["matched_documents"] = interaction_dict["matched_documents"].T
        if len(interaction_dict["matched_documents"]) > 0:
            for k in interaction_dict["matched_documents"].keys():
                interaction_dict[f"matched_documents_{k}"] = interaction_dict["matched_documents"][k].values
        del interaction_dict["matched_documents"]

        return interaction_dict

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

        if self.form is not None:
            to_encode["form"] = self.form.to_json()

        return jsonable_encoder(to_encode, custom_encoder=custom_encoder)

    @classmethod
    def from_dict(cls, interaction_dict: dict, feedback_cls: Optional[Type[StandardForm]] = None) -> Interaction:
        # remove the _id from mongodb
        if "_id" in interaction_dict.keys():
            del interaction_dict["_id"]

        interaction_dict["user_completions"] = [Completion.from_dict(r) for r in interaction_dict["user_completions"]]

        if "form" in interaction_dict.keys():
            # The interaction contained a type of form, e.g. feedback form, parse it accordingly

            # Make sure the user specified a feedback_cls
            assert feedback_cls is not None, "You must specify which type of feedback it is"

            interaction_dict["form"] = feedback_cls.from_dict(interaction_dict["form"])

        return cls(**interaction_dict)


def read_collection(
    mongo_db: pymongo.database.Database,
    collection: str,
    feedback_cls: Optional[Type[StandardForm]] = None,
    filters: Optional[dict] = None,
) -> pd.DataFrame:
    """Read a collection from mongodb.

    Returns the data in a dataframe for convenience.

    If the collection is an instance of Interaction, no feedback_cls is required. However, if a form is attached, i.e. entry["form"]
    By default, return all items in the collection. If filters are provided, return only items that matches the filters.
    For example, to get just the interactions from a specific user, use filters={"username": <username>}.
    """
    flattened_interactions = []
    skipped_interactions = []
    interactions = mongo_db[collection].find(filters)
    for interaction in interactions:
        try:
            flattened_interaction = Interaction.from_dict(interaction, feedback_cls=feedback_cls).flatten()
            flattened_interactions.append(flattened_interaction)
        except Exception as err:
            skipped_interactions.append(interaction)

    logger.info(f"Retrieved {len(flattened_interactions)} entries. Skipped {len(skipped_interactions)} entries")

    return pd.DataFrame(flattened_interactions)
