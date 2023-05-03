"""Performance tests for the bot.

We have 3 categories of questions:
- Relevant
- Irrelevant
- Trick

We want to test the following:
- The bot should answer relevant questions
- The bot should NOT answer irrelevant questions
- The bot should NOT answer trick questions

We detect whether the bot answered or not using the unknown embedding.
As artifacts, we generate two files:
- results_detailed.csv: contains the questions, the answers and whether it was judged relevant or not.
- results_summary.csv: aggregated results per category.
"""
import logging
import random

import pandas as pd
import pytest
from buster.busterbot import Buster

from src import cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


EMBEDDING_LENGTH = 1536


@pytest.fixture
def busterbot(monkeypatch, run_expensive):
    if not run_expensive:
        random.seed(42)

        # Patch embedding call
        monkeypatch.setattr(
            Buster, "get_embedding", lambda s, x, engine: [random.random() for _ in range(EMBEDDING_LENGTH)]
        )

    buster = Buster(cfg=cfg.buster_cfg, retriever=cfg.retriever)
    return buster


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("results", numbered=False)


def process_questions(busterbot, questions: list[str]) -> pd.DataFrame:
    results = []
    for question in questions:
        result = busterbot.process_input(question)
        results.append((question, result.is_relevant, result.completion.text))

    return pd.DataFrame(results, columns=["question", "is_relevant", "answer"])


@pytest.mark.parametrize(
    "source_file,target_file",
    [
        ("src/Questions dataset - Relevant.csv", "relevant_questions.csv"),
        ("src/Questions dataset - Irrelevant.csv", "irrelevant_questions.csv"),
        ("src/Questions dataset - Trick.csv", "trick_questions.csv"),
    ],
)
def test_questions(results_dir, busterbot, source_file, target_file):
    questions = pd.read_csv(source_file, header=None)[0].to_list()

    results = process_questions(busterbot, questions)
    results.to_csv(results_dir / target_file, index=False)


@pytest.mark.order("last")
def test_summary(results_dir):
    results = pd.concat(
        [
            pd.read_csv(results_dir / "relevant_questions.csv"),
            pd.read_csv(results_dir / "irrelevant_questions.csv"),
            pd.read_csv(results_dir / "trick_questions.csv"),
        ],
        keys=["Relevant", "Irrelevant", "Trick"],
        names=["Category", "index"],
    )
    results.reset_index().to_csv("results_detailed.csv", index=False)
    results.drop(columns=["answer"], inplace=True)

    logger.info(results.head())
    summary = results.groupby("Category").agg({"is_relevant": ["sum", "count"]}).reset_index()
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary.rename(columns={"Category_": "Category"}, inplace=True)
    column_order = ["Category"] + [col for col in summary.columns if col not in ["Category"]]
    summary = summary.loc[:, column_order]

    summary["Score"] = summary.apply(lambda x: f"{x['is_relevant_sum']} / {x['is_relevant_count']}", axis=1)
    summary["Score (%)"] = (summary["is_relevant_sum"] / summary["is_relevant_count"]).apply(
        lambda x: f"{x * 100:04.2f} %"
    )
    summary.drop(columns=["is_relevant_sum", "is_relevant_count"], inplace=True)
    summary.set_index("Category", inplace=True)

    logger.info(summary)
    summary.to_csv("results_summary.csv")
