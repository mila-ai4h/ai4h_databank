import logging
import os
import re

import openai
import pandas as pd

from buster.completers import ChatGPTCompleter, UserInputs
from src.cfg import retriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# set openai creds
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")


def split_questions(text: str):
    """Splits a raw text containing multiple questions into separate questions.


    Args:
        text (str): The raw text containing multiple questions.

    Returns:
        list: A list of individual questions extracted from the input text.

    Example:
        >>> text = '1) What is your name?\n2) How old are you?\n3) Where do you live?\n'
        >>> split_questions(text)
        ['What is your name?', 'How old are you?', 'Where do you live?']

    """
    # This pattern matches the question number followed by ') ' and captures the rest of the line as the question text. Thanks ChatGPT
    pattern = r"\d+\)\s+(.*)"
    questions = re.findall(pattern, text)
    return questions


def generate_questions(document: str, completer):
    """Generates questions based on the provided documents using a language model."""

    prompt = """You generate questions based on the documents provided by the user.
    Questions should be in the style of a typical person interested in AI policies.
    They can be general, or specific, but should be answered by the documents provided.
    Generate 3 questions, keep them short and on-topic.
    RULES:
    1) Avoid using terms directly from the documents.
    2) Don't refer directly to the document, for example, do not say 'According to the document'.
    3) Questions should be the kind of information you'd search for on a website, and generally broad in scope.
    """
    outputs = completer.complete(prompt, UserInputs(original_input=document), **completer.completion_kwargs)

    # at this point, chatGPT generated questions in form 1) ... 2) ... so we split them
    questions = split_questions(outputs)
    print(f"{questions=}")
    return questions


def generate_questions_from_summary(document: str, completer):
    """Generates a summary of a document, then generates a question using a language model."""

    # summarize the document
    prompt = "Summarize the content of the following document. Keep it high level"
    summary = completer.complete(prompt, UserInputs(original_input=document), **completer.completion_kwargs)
    print(f"{summary=}")

    # generate questions
    prompt = """You generate questions based on the documents provided by the user.
    Questions should be in the style of a typical person interested in AI policies.
    They can be general, or specific, but should be answered by the information provided.
    Generate 3 questions, keep them short and on-topic."""
    outputs = completer.complete(prompt, UserInputs(original_input=summary), **completer.completion_kwargs)
    print(f"{outputs=}")
    outputs = completer.complete(prompt, UserInputs(original_input=document), **completer.completion_kwargs)

    # at this point, chatGPT generated questions in form 1) ... 2) ... so we split them
    questions = split_questions(outputs)
    return questions, summary


def main():
    # to declare beforehand
    documents_file = "documents.csv"  # csv containing all chunks, or documents, to draw questions from
    outputs_file = "questions.csv"  # Where to save the generated questions to
    num_documents = 50  # Number of documents to use (generates 3 questions per doc)
    model = "gpt-4"  # chatGPT model to use

    if not os.path.isfile(documents_file):
        # If the document isn't already found localoly, it will fetch it from mongodb and save a local copy
        df = retriever.get_documents()
        df = df.sort_values(by=["source", "title", "country", "year"])
        df.to_csv(documents_file, index=False)

    # load the df, then shuffle it
    df = pd.read_csv(documents_file)
    df = df.sample(frac=1, random_state=42)

    # used for completions
    completer = ChatGPTCompleter(
        documents_formatter=None,
        prompt_formatter=None,
        completion_kwargs={"model": model, "stream": False, "temperature": 0},
    )

    results = {
        "document": [],
        "question": [],
        # 'summary': [],
    }

    for idx in range(num_documents):
        # get a document
        document = df.iloc[idx].content
        print(f"{document=}")

        questions = generate_questions(document, completer)

        for question in questions:
            results["document"].append(document)
            results["question"].append(question)

    results_df = pd.DataFrame(results)
    results_df.to_csv(outputs_file, index=False)


if __name__ == "__main__":
    main()
