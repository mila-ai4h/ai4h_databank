import copy
import logging
import os

import openai
import pandas as pd
from tqdm import tqdm

from buster.busterbot import Buster
from src import cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

openai.organization = os.getenv("OPENAI_ORGANIZATION")

# for iterating over df
tqdm.pandas()


buster_cfg = copy.deepcopy(cfg.buster_cfg)
buster = Buster(completer=cfg.completer, retriever=cfg.retriever, validator=cfg.validator)


def process_questions(q):
    try:
        # gets a completion
        completion = buster.process_input(q.question)

        q["answer_relevant"] = completion.answer_relevant
        q["question_relevant"] = completion.question_relevant
        q["answer"] = completion.answer_text
        q["error"] = completion.error

    except Exception as e:
        logger.exception("something went wrong...")
        q["answer_relevant"] = None
        q["question_relevant"] = None
        q["answer"] = None
        q["error"] = True
    return q


def is_correct(q) -> pd.Series:
    question_type = q.question_type
    if question_type == "relevant":
        expected_question_relevant = True
        expected_answer_relevant = True
    elif question_type == "irrelevant":
        expected_question_relevant = False
        expected_answer_relevant = False
    elif question_type == "trick":
        expected_question_relevant = True
        expected_answer_relevant = False
    else:
        raise ValueError(f"Invalid {question_type=}")

    if q.error:
        q["is_correct"] = False

    else:
        q["is_correct"] = (q.question_relevant == expected_question_relevant) & (
            q.answer_relevant == expected_answer_relevant
        )

    return q


if __name__ == "__main__":
    questions_df = pd.read_csv("../sample_questions.csv")

    # feed questions to Buster and evaluate relevance
    results_df = questions_df.progress_apply(process_questions, axis=1)

    # based on results, compute if it's correct or not
    results_df = results_df.progress_apply(is_correct, axis=1)

    # save results to csv
    results_df.to_csv("question_answer_results.csv", index=False)

    for question_type in ["relevant", "irrelevant", "trick"]:
        # filter out by question type and remove those that might have had an error...
        sub_df = results_df[(results_df.question_type == question_type) & (results_df.error == False)]
        print(f"Result for {question_type=}: {sum(sub_df.is_correct)}/{len(sub_df.is_correct)}")
