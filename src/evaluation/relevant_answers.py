import copy
import logging
import os

import openai
import pandas as pd
from buster.busterbot import Buster
from tqdm import tqdm

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


def compute_result(results_df, question_type: str):
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

    # filter by question type
    sub_df = results_df[results_df.question_type == question_type]
    # drop entries with errors
    sub_df = sub_df[sub_df.error == False]

    # compute stats
    num_correct = sum(
        (sub_df.question_relevant == expected_question_relevant) & (sub_df.answer_relevant == expected_answer_relevant)
    )
    total = len(sub_df)

    return num_correct, total


if __name__ == "__main__":
    questions_df = pd.read_csv("../sample_questions.csv")
    results_df = questions_df.progress_apply(process_questions, axis=1)

    # save results to csv
    results_df.to_csv("question_results.csv", index=False)

    for question_type in ["relevant", "irrelevant", "trick"]:
        num_correct, total = compute_result(results_df, question_type)
        print(f"Result for {question_type=}: {num_correct}/{total}")
