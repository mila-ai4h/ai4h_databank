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
from pinecone.core.exceptions import PineconeProtocolError
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from buster.busterbot import Buster
from buster.completers import ChatGPTCompleter, DocumentAnswerer
from buster.formatters.documents import DocumentsFormatterJSON
from buster.formatters.prompts import PromptFormatter
from buster.retriever import ServiceRetriever
from buster.tokenizers import GPTTokenizer
from buster.validators import Validator
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
        monkeypatch.setattr(Validator, "check_question_relevance", lambda s, q: (True, "mocked response"))
        # Patch embedding call to avoid computing embeddings
        monkeypatch.setattr(
            ServiceRetriever, "get_embedding", lambda s, x, model: [random.random() for _ in range(EMBEDDING_LENGTH)]
        )
        # set thresh = 1 to be sure that no documents get retrieved
        buster_cfg.retriever_cfg["thresh"] = 1

    retriever = ServiceRetriever(**buster_cfg.retriever_cfg)

    tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    document_answerer: DocumentAnswerer = DocumentAnswerer(
        completer=ChatGPTCompleter(**buster_cfg.completion_cfg),
        documents_formatter=DocumentsFormatterJSON(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
        **buster_cfg.documents_answerer_cfg,
    )
    validator = Validator(**buster_cfg.validator_cfg)

    buster: Buster = Buster(retriever=retriever, document_answerer=document_answerer, validator=validator)

    return buster


def process_questions(busterbot, questions: pd.DataFrame) -> pd.DataFrame:
    def is_unable_to_process(answer: pd.Series) -> bool:
        return answer.answer_text.startswith("Unable to process your question at the moment, try again soon")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry=(retry_if_result(is_unable_to_process) | retry_if_exception_type(PineconeProtocolError)),
    )
    def answer_question(question):
        completion = busterbot.process_input(question.question)
        if "title" in completion.matched_documents.columns:
            sources_titles = completion.matched_documents.title.tolist()
        else:
            sources_titles = ["" for _ in range(3)]
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


def compute_summary(markdown_summary: str, results: pd.DataFrame) -> str:
    """
    Compute a summary of relevant, irrelevant and trick questions and answers from the results DataFrame and append it to a markdown summary.

    This function groups the results by question type and calculates the sum and count of relevant questions and answers.
    It then formats this summary into a readable format, converts it to markdown, and appends it to the provided markdown summary.
    The final markdown summary is returned.

    Parameters:
    - markdown_summary: The initial markdown summary to which the computed summary will be appended.
    - results: A DataFrame containing the results to be summarized. Expected columns are ["question_type", "question_relevant", "answer_relevant"].

    Returns:
    - str: The markdown summary with the appended computed summary.

    Note:
    - The resulting summary is saved to a CSV file named "results_summary.csv" and logged using the logger.
    """
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

    markdown_summary += "# Performance Results\n\n"
    markdown_summary += summary.to_markdown(tablefmt="github")
    markdown_summary += "\n\n"
    markdown_summary += "## Relevant\n Questions that would be realistically asked by policy makers and lawyers, and whose answer should be in our knowledge base.\nBoth the question and the answer should be relevant.\n**Example**: What is the focus of Italy's AI policy strategy?\n\n"
    markdown_summary += "This category is further divided in 2: original and variants. Originals are questions provided by the OECD. Variants are questions generated by GPT. The goal of the variants is to measure how sensitive the system is to the specific phrasing used. They are studied in more details in the Robustness section below.\n"
    markdown_summary += "## Irrelevant\n Questions that are out of scope.\nBoth the question and the answer should be irrelevant.\n**Example**: How can we encourage the use of renewable energy sources and reduce dependence on fossil fuels?\n"
    markdown_summary += "## Trick\n Questions that could realistically be asked, but that the model cannot answer.\nThe question should be marked as relevant, but the answer as irrelevant.\n**Example**: Tell me about [made up AI policy].\n"
    markdown_summary += "\n\n"

    return markdown_summary


def detect_according_to_the_documentation(markdown_summary: str, results: pd.DataFrame) -> str:
    """
    Detect and summarize the usage of forbidden expressions in the answers from the results DataFrame and append it to a markdown summary.

    This function checks each answer in the results DataFrame for the presence of specified forbidden expressions.
    It calculates the total number of occurrences and appends this information, along with some explanatory text,
    to the provided markdown summary. The final markdown summary is returned.

    Parameters:
    - markdown_summary: The initial markdown summary to which the computed summary will be appended.
    - results: A DataFrame containing the results to be checked. Expected to contain a column named "answer_text".

    Returns:
    - str: The markdown summary with the appended summary of forbidden expressions usage.

    Note:
    - The computed summary includes the count and percentage of answers containing forbidden expressions.
    - The markdown summary is appended with a header and explanatory text regarding the detector and forbidden expressions.
    """
    forbidden_expressions = [
        "according to the documentation",
        "based on the documentation",
        "the provided documents",
    ]

    def detect_forbidden_expression(answer: str) -> bool:
        return any([expr in answer.lower() for expr in forbidden_expressions])

    fail = sum(results.answer_text.apply(detect_forbidden_expression))
    total = len(results)

    markdown_summary += "# Expressions Detector\n\n"
    markdown_summary += "This detector checks whether the system used expressions we want to discourage.\n"
    markdown_summary += f"- **According to the documentation**: {fail} / {total} ({fail / total * 100:04.2f} %)\n"
    markdown_summary += (
        "    - Include also the following variants: 'based on the documentation', 'the provided documents'\n"
    )
    markdown_summary += "\n\n"

    return markdown_summary


def measure_robustness(markdown_summary: str, results: pd.DataFrame) -> str:
    """
    Measure and summarize the robustness of relevant questions and answers in the results DataFrame and append it to a markdown summary.

    This function filters the results for relevant questions, groups them, and calculates the sum of relevant questions and answers.
    It then formats this summary into a readable format, converts it to markdown, and appends it to the provided markdown summary.
    The final markdown summary is returned.

    Parameters:
    - markdown_summary: The initial markdown summary to which the computed summary will be appended.
    - results: A DataFrame containing the results to be summarized. Expected columns are ["question_type", "group", "question_relevant", "answer_relevant"].

    Returns:
    - str: The markdown summary with the appended computed robustness summary.
    """
    results = results[results.question_type.str.startswith("relevant")]
    grouped = results.groupby("group")["question_relevant"].sum().reset_index()
    question_counts = grouped["question_relevant"].value_counts().sort_index().rename("Relevant questions")

    grouped = results.groupby("group")["answer_relevant"].sum().reset_index()
    answer_counts = grouped["answer_relevant"].value_counts().sort_index().rename("Relevant answers")

    counts = pd.concat([question_counts, answer_counts], axis=1).T
    max_val = results.groupby("group").size().iloc[0]
    for i in range(max_val + 1):
        if i not in counts.columns:
            counts[i] = 0
    counts = counts.reindex(sorted(counts.columns), axis=1)

    counts.rename(columns=lambda x: f"{x} / {max_val}", inplace=True)
    counts.fillna(0, inplace=True)

    markdown_summary += "# Robustness\n\n"
    markdown_summary += "Each relevant question has 4 variants. The model should behave similarly for all variants. Variants were generated to have different levels of fluency in English.\n"
    markdown_summary += "## Relevance Robustness\n"
    markdown_summary += "This is the distribution of relevance for both questions and answers. For example, 'Relevant questions' at 3 / 5 means how many groups had 3 / 5 questions judged relevant.\n"
    markdown_summary += counts.to_markdown(tablefmt="github")
    markdown_summary += "\n\n"

    return markdown_summary


def evaluate_performance(busterbot):
    """
    Evaluate the performance of the system on a sample of questions and generate a markdown summary.

    This function reads a sample of questions from a CSV file, processes them, and evaluates the results
    using three different metrics: a general summary, detection of forbidden expressions, and robustness measurement.
    The evaluations are appended to a markdown summary, which is then written to a markdown file.

    Parameters:
    - busterbot: An instance of buster that can process questions.

    Note:
    - The sample questions should be stored in a CSV file named "sample_questions_variants.csv" in the "data" directory.
    - The resulting markdown summary is saved to a file named "results_summary.md".
    """
    questions = pd.read_csv("data/sample_questions_variants.csv")
    results = process_questions(busterbot, questions)

    markdown_summary = ""
    markdown_summary = compute_summary(markdown_summary, results)
    markdown_summary = detect_according_to_the_documentation(markdown_summary, results)
    markdown_summary = measure_robustness(markdown_summary, results)

    with open("results_summary.md", "w") as f:
        f.write(markdown_summary)


def test_summary(busterbot):
    # This is necessary to test evaluate_performance, otherwise pytest try to run it as a test.
    # Long term fix is to not use pytest for performance tests.
    evaluate_performance(busterbot)
