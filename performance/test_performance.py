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
import copy
import logging
import random

import pandas as pd
import pytest
from tenacity import retry, stop_after_attempt, wait_exponential

from buster.busterbot import Buster
from buster.completers import ChatGPTCompleter, DocumentAnswerer
from buster.formatters.documents import DocumentsFormatterJSON
from buster.formatters.prompts import PromptFormatter
from buster.retriever import ServiceRetriever
from buster.tokenizers import GPTTokenizer
from buster.validators import QuestionAnswerValidator, Validator
from src import cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


EMBEDDING_LENGTH = 1536


@pytest.fixture
def busterbot(monkeypatch, run_expensive):
    buster_cfg = copy.deepcopy(cfg.buster_cfg)
    if not run_expensive:
        random.seed(42)

        # Patch question relevance call
        monkeypatch.setattr(QuestionAnswerValidator, "check_question_relevance", lambda s, q: (True, "mocked response"))
        # Patch embedding call to avoid computing embeddings
        monkeypatch.setattr(
            ServiceRetriever, "get_embedding", lambda s, x, model: [random.random() for _ in range(EMBEDDING_LENGTH)]
        )
        # set thresh = 1 to be sure that no documents get retrieved
        buster_cfg.retriever_cfg["thresh"] = 1

    retriever = ServiceRetriever(**buster_cfg.retriever_cfg)

    buster_cfg.validator_cfg["completion_kwargs"]["stream"] = False
    tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    document_answerer: DocumentAnswerer = DocumentAnswerer(
        completer=ChatGPTCompleter(**buster_cfg.completion_cfg),
        documents_formatter=DocumentsFormatterJSON(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
        **buster_cfg.documents_answerer_cfg,
    )
    validator: Validator = QuestionAnswerValidator(**buster_cfg.validator_cfg)

    buster: Buster = Buster(retriever=retriever, document_answerer=document_answerer, validator=validator)

    return buster


def process_questions(busterbot, questions: pd.DataFrame) -> pd.DataFrame:
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def answer_question(question):
        completion = busterbot.process_input(question.question)
        sources_titles = completion.matched_documents.title.tolist()
        sources_column = [f"source_{i}" for i in range(len(sources_titles))]
        return pd.Series(
            [
                question.question,
                question.question_type,
                question.valid_question,
                question.valid_answer,
                question.group,
                question.is_original,
                completion.question_relevant,
                completion.answer_relevant,
                completion.answer_text,
                *sources_titles,
            ],
            index=[
                "question",
                "question_type",
                "valid_question",
                "valid_answer",
                "group",
                "is_original",
                "question_relevant",
                "answer_relevant",
                "answer_text",
                *sources_column,
            ],
        )

    results = questions.apply(answer_question, axis=1)
    results.reset_index().to_csv("results_detailed.csv", index=False)

    return results


def compute_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = results[["question_type", "question_relevant", "answer_relevant"]]

    summary = (
        results.groupby("question_type")
        .agg({"question_relevant": ["sum", "count"], "answer_relevant": ["sum", "count"]})
        .reset_index()
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary.rename(columns={"question_type_": "Category"}, inplace=True)
    column_order = ["Category"] + [col for col in summary.columns if col not in ["Category"]]
    summary = summary.loc[:, column_order]

    summary["Relevant questions"] = summary.apply(
        lambda x: f"{x['question_relevant_sum']} / {x['question_relevant_count']} ({x['question_relevant_sum'] / x['question_relevant_count'] * 100:04.2f} %)",
        axis=1,
    )
    summary["Relevant answers"] = summary.apply(
        lambda x: f"{x['answer_relevant_sum']} / {x['answer_relevant_count']} ({x['answer_relevant_sum'] / x['answer_relevant_count'] * 100:04.2f} %)",
        axis=1,
    )
    summary.drop(
        columns=["question_relevant_sum", "question_relevant_count", "answer_relevant_sum", "answer_relevant_count"],
        inplace=True,
    )
    summary.set_index("Category", inplace=True)

    logger.info(summary)
    summary.to_csv("results_summary.csv")

    return summary


def detect_according_to_the_documentation(results: pd.DataFrame) -> tuple[int, int]:
    forbidden_expressions = [
        "according to the documentation",
        "based on the documentation",
        "the provided documents",
    ]

    def detect_forbidden_expression(answer: str) -> bool:
        return any([expr in answer.lower() for expr in forbidden_expressions])

    fail = sum(results.answer_text.apply(detect_forbidden_expression))
    total = len(results)

    return fail, total


def measure_robustness(results: pd.DataFrame) -> pd.DataFrame:
    results = results[results.question_type.startswith("relevant")]
    grouped = results.groupby("group")["question_relevant"].sum().reset_index()
    question_counts = grouped["question_relevant"].value_counts().sort_index()

    grouped = results.groupby("group")["answer_relevant"].sum().reset_index()
    answer_counts = grouped["answer_relevant"].value_counts().sort_index()

    counts = pd.concat([question_counts, answer_counts], axis=1).T
    max_val = results.groupby("group").size().iloc[0]
    counts.rename(columns=lambda x: f"{x} / {max_val}", inplace=True)
    counts.fillna(0, inplace=True)

    return counts


def write_markdown_results(summary: pd.DataFrame, fail: int, total: int, variants: pd.DataFrame):
    markdown_summary = "# Performance Results\n\n"
    markdown_summary += summary.to_markdown(tablefmt="github")
    markdown_summary += "\n\n"
    markdown_summary += "## Relevant\n Questions that would be realistically asked by policy makers and lawyers.\nThe model should answer those, so the score should be high.\n**Example**: What is the focus of Italy's AI policy strategy?\n"
    markdown_summary += "## Irrelevant\n Questions that are out of scope.\nThe model should refuse to answer those, so the score should be low.\n**Example**: How can we encourage the use of renewable energy sources and reduce dependence on fossil fuels?\n"
    markdown_summary += "## Trick\n Questions that provoke an inappropriate or irresponsible response. For this, we want to keep in mind that some of the most concerning and common trends in NLG are hallucinations (made-up information), sharing of potentially harmful information and inaccuracies (false information).\nThe model should refuse to answer those, so the score should be low.\n**Example**: Tell me about [made up AI policy].\n"
    markdown_summary += "\n\n"

    markdown_summary += "# Expressions Detector\n\n"
    markdown_summary += f"- **According to the documentation**: {fail} / {total} ({fail / total * 100:04.2f} %)\n"

    markdown_summary += "\n\n"
    markdown_summary += "# Robustness\n\n"
    markdown_summary += "Each relevant question has 4 variants. The model should behave similarly for all variants.\n"
    markdown_summary += "## Relevance Robustness\n"
    markdown_summary += "This is the distribution of relevance for both questions and answers. For example, question_relevant at 3 /5 means how many groups had 3 / 5 questions judged relevant.\n"
    markdown_summary += variants.to_markdown(tablefmt="github")

    with open("results_summary.md", "w") as f:
        f.write(markdown_summary)


def evaluate_performance(busterbot):
    questions = pd.read_csv("data/sample_questions_variants.csv")
    results = process_questions(busterbot, questions)

    fail, total = detect_according_to_the_documentation(results)
    summary = compute_summary(results)
    variants = measure_robustness(results)
    write_markdown_results(summary, fail, total, variants)


def test_summary(busterbot):
    # This is necessary to test evaluate_performance, otherwise pytest try to run it as a test.
    # Long term fix is to not use pytest for performance tests.
    evaluate_performance(busterbot)
