import json
import os
from copy import deepcopy

import pandas as pd
from tqdm.contrib.concurrent import thread_map

from buster.validators.validators import QuestionValidator


def parse_jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Parses a JSONL file into a pandas DataFrame.

    Args:
    file_path (str): The path to the JSONL file.

    Returns:
    pd.DataFrame: A DataFrame containing the parsed data.
    """
    data = []
    with open(file_path, "r") as file:
        for line in file:
            json_line = json.loads(line)

            # extract data line by line
            user_input = json_line["messages"][1]["content"]
            annotation = json_line["messages"][2]["content"]

            # convert it to bool
            annotation = annotation == "true"

            data.append({"user_input": user_input, "annotation": annotation})

    return pd.DataFrame(data)


# Example usage

question_validator_cfg = {
    "invalid_question_response": """Thank you for your question! Unfortunately, I haven't been able to find the information you're looking for. Your question might be:
            * Outside the scope of AI policy documents
            * Too recent (i.e. draft policies) or about the future
            * Building on my previous answer (I have no memory of previous conversations)
            * Vague (i.e not affiliated with a specific country)
            * Asking the model to perform its own assessment of the policies (i.e. What is the best/worst AI policy)
            You can always try rewording your question and ask again!
            """,
    "check_question_prompt": """You are a chatbot answering questions on behalf of the OECD specifically on AI policies.
Your first job is to determine whether or not a question is valid, and should be answered.
For a question to be considered valid, it must be related to AI and policies.
More general questions are not considered valid, even if you might know the response.
A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.
Do not judge the tone of the question. As long as it is relevant to the topic, respond 'true'.

For example:
Q: What policies did countries like Canada put in place with respect to artificial intelligence?
true

Q: What policies are put in place to ensure the wellbeing of agriculture?
false

Q:
""",
    "completion_kwargs": {
        "stream": False,
        "temperature": 0,
    },
    "client_kwargs": {
        "timeout": 10,
        "max_retries": 2,
    },
}

# if __name__ == "__main__":

# Load the test dataset
df = parse_jsonl_to_dataframe("test_dataset.jsonl")
print(f"Number of test sampels: {len(df)}")

# Load the finetuned model
cfg_ft = deepcopy(question_validator_cfg)
cfg_ft["completion_kwargs"]["model"] = "ft:gpt-3.5-turbo-0613:oecd-ai:first-finetune:8LEyi8pG"
qv_ft = QuestionValidator(**cfg_ft)

# Load the original model
cfg_orig = deepcopy(question_validator_cfg)
cfg_orig["completion_kwargs"]["model"] = "gpt-3.5-turbo-0613"
qv_orig = QuestionValidator(**cfg_orig)


# Compute predictions of original model
outputs = thread_map(qv_orig.check_question_relevance, df.user_input.to_list(), max_workers=4)
preds = [output[0] for output in outputs]  # Extract results
df["prediction_orig"] = preds

# Compute predictions of finetuned model
outputs = thread_map(qv_ft.check_question_relevance, df.user_input.to_list(), max_workers=4)
preds = [output[0] for output in outputs]  # Extract results
df["prediction_ft"] = preds

# Compute accuracy
acc_orig = sum(df.prediction_orig == df.annotation) / len(df)
acc_ft = sum(df.prediction_ft == df.annotation) / len(df)

print(f"Accuracy using vanilla gpt: {acc_orig:2.2%}")
print(f"Accuracy using finetuned gpt: {acc_ft:2.2%}")
