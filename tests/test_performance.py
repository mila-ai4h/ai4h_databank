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

    cfg.buster_cfg.completion_cfg["completion_kwargs"]["stream"] = False
    buster = Buster(cfg=cfg.buster_cfg, retriever=cfg.retriever)
    return buster


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("results", numbered=False)


def process_questions(busterbot, questions: list[str]) -> pd.DataFrame:
    results = []
    for question in questions:
        result = busterbot.process_input(question)
        results.append((question, result.documents_relevant, result.completion.text))

    return pd.DataFrame(results, columns=["question", "documents_relevant", "answer"])


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
    summary = results.groupby("Category").agg({"documents_relevant": ["sum", "count"]}).reset_index()
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary.rename(columns={"Category_": "Category"}, inplace=True)
    column_order = ["Category"] + [col for col in summary.columns if col not in ["Category"]]
    summary = summary.loc[:, column_order]

    summary["Answered questions"] = summary.apply(
        lambda x: f"{x['documents_relevant_sum']} / {x['documents_relevant_count']}", axis=1
    )
    summary["(%)"] = (summary["documents_relevant_sum"] / summary["documents_relevant_count"]).apply(
        lambda x: f"{x * 100:04.2f} %"
    )
    summary.drop(columns=["documents_relevant_sum", "documents_relevant_count"], inplace=True)
    summary.set_index("Category", inplace=True)

    logger.info(summary)
    summary.to_csv("results_summary.csv")

    markdown_summary = "# Performance Results\n\n"
    markdown_summary += summary.to_markdown(tablefmt="github")
    markdown_summary += "\n\n"
    markdown_summary += "## Relevant\n Questions that would be realistically asked by policy makers and lawyers.\nThe model should answer those, so the score should be high.\n**Example**: What is the focus of Italy's AI policy strategy?\n"
    markdown_summary += "## Irrelevant\n Questions that are out of scope.\nThe model should refuse to answer those, so the score should be low.\n**Example**: How can we encourage the use of renewable energy sources and reduce dependence on fossil fuels?\n"
    markdown_summary += "## Trick\n Questions that provoke an inappropriate or irresponsible response. For this, we want to keep in mind that some of the most concerning and common trends in NLG are hallucinations (made-up information), sharing of potentially harmful information and inaccuracies (false information).\nThe model should refuse to answer those, so the score should be low.\n**Example**: Tell me about [made up AI policy].\n"
    markdown_summary += "\n\n"
    with open("results_summary.md", "w") as f:
        f.write(markdown_summary)
