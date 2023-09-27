from ast import literal_eval

import openai
import pandas as pd
from tqdm import tqdm


def create_question_variants(question: str, n_variant: int = 4) -> list[str]:
    """
    Generate multiple variants of a given question.

    Parameters:
    - question: the original question for which variants are to be generated.
    - n_variant: the number of variants to generate. Default is 4.

    Returns:
    - list[str]: A list containing the variants of the original question.

    Raises:
    - Exception: If OpenAI does not stop generating variants.
    - ValueError: If the model does not generate a valid list or the correct number of variants.
    """
    messages = [
        {
            "role": "system",
            "content": f"Rephrase the following question in {n_variant} different ways as a Python list:",
        },
        {"role": "user", "content": question},
    ]
    response = openai.ChatCompletion.create(messages=messages, model="gpt-4-0613", temperature=0)

    if response.choices[0].finish_reason != "stop":
        raise Exception(
            f"Something went wrong. OpenAI did not stop generating variants. finish_reason = {response.choices[0].finish_reason}"
        )

    variants = literal_eval(response.choices[0].message.content)

    if not isinstance(variants, list):
        raise ValueError(f"The model did not generate a valid list. Got: {variants}")

    if len(variants) != n_variant:
        raise ValueError(
            f"The model did not generate the correct number of variants. Got: {len(variants)}, expected: {n_variant}"
        )

    return variants


def create_all_variants(questions: pd.DataFrame, n_variant: int = 4) -> pd.DataFrame:
    """
    Generate multiple variants for each question in a DataFrame.

    Parameters:
    - questions: DataFrame containing the original questions.
    - n_variant: the number of variants to generate for each question. Default is 4.

    Returns:
    - pd.DataFrame: A DataFrame containing the original questions and their variants.

    Note:
    - The input DataFrame should have a column named 'question' containing the questions.
    - The output DataFrame will have a column named 'group' that has the same value for a question and its variants.
    """

    def generate_variants(row):
        if row.question_type != "relevant":
            new_df = pd.concat([row], axis=1).T
            new_df["group"] = row.name
            new_df["is_original"] = True
        else:
            new_df = pd.concat([row] * (n_variant + 1), axis=1).T

            variants = create_question_variants(row.question, n_variant)
            new_df["question"] = [row.question] + variants
            new_df["question_type"] = ["relevant (original)"] + ["relevant (variant)"] * n_variant

            new_df["group"] = row.name
            new_df["is_original"] = [True] + [False] * n_variant

        return new_df

    tqdm.pandas()
    questions = pd.concat(questions.progress_apply(generate_variants, axis=1).tolist(), ignore_index=True)

    return questions


if __name__ == "__main__":
    questions = pd.read_csv("src/sample_questions.csv")
    questions = create_all_variants(questions, n_variant=4)
    questions.to_csv("src/sample_questions_variants.csv", index=False)
