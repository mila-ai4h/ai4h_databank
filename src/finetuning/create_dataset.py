import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI

client = OpenAI(organization=os.environ["OPENAI_ORGANIZATION"])

encoding = tiktoken.get_encoding("cl100k_base")

system_prompt = """You are a chatbot answering questions on behalf of the OECD specifically on AI policies.
Your first job is to determine whether or not a question is valid, and should be answered.
For a question to be considered valid, it must be related to AI and policies.
More general questions are not considered valid, even if you might know the response.
A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.
Do not judge the tone of the question. As long as it is relevant to the topic, respond 'true'.

Q: """


def format_data(prompt, question, ground_truth):
    data = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": ground_truth},
        ]
    }
    return data


def format_data_df(x):
    return format_data(system_prompt, x.completion_user_input, x.ground_truth)


def series_to_jsonl(series: pd.Series, filename: str) -> None:
    """
    Converts a Pandas Series to a JSON Lines (JSONL) file.

    Args:
    series (pd.Series): The Pandas Series to convert.
    filename (str): The name of the file to write the JSONL data to.
    """
    with open(filename, "w") as file:
        for item in series.items():
            # Convert each item to JSON and write to file
            json.dump(item[1], file)
            file.write("\n")


def preprocess_df(df):
    df = df.dropna()
    df = df[df.Annotation != "Other"]
    df["ground_truth"] = df.Annotation.apply(lambda x: "true" if x == "Relevant" else "false")
    return df


def load_dataset(data_path):
    # Load the dataset
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
    return dataset


def check_for_errors(dataset):
    "Taken from : https://cookbook.openai.com/examples/chat_finetuning_data_prep"
    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)

    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
        raise ValueError("Address found errors.")
    else:
        print("No errors found")


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    "Taken from : https://cookbook.openai.com/examples/chat_finetuning_data_prep"
    # not exact!
    # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    "Taken from : https://cookbook.openai.com/examples/chat_finetuning_data_prep"
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def print_distribution(values, name):
    "Taken from : https://cookbook.openai.com/examples/chat_finetuning_data_prep"
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


def token_count_stats(dataset):
    "Taken from : https://cookbook.openai.com/examples/chat_finetuning_data_prep"
    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > 4096 for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")


def split_train_test(df, train_pct=0.8, seed=42):
    # shuffle
    shuffled_df = df.sample(len(df), random_state=seed)

    pivot = int(train_pct * len(df))
    df_train, df_test = shuffled_df.iloc[0:pivot], shuffled_df.iloc[pivot:]

    print(f"Number of samples in train: {len(df_train)}")
    print(f"Number of samples in test: {len(df_test)}")

    return df_train, df_test


def upload_finetune_dataset(data_path):
    print(f"Uploading {data_path} for fine-tuning")
    client.files.create(file=open(data_path, "rb"), purpose="fine-tune")


def dataset_stats(df):
    print()
    print("Dataset distribution:")
    print(df.ground_truth.value_counts())
    print()


if __name__ == "__main__":
    annotation_file = "./annotations_jeremy.csv"

    df = pd.read_csv(annotation_file)
    df = preprocess_df(df)  # remove nans, set ground truth

    train_df, test_df = split_train_test(df, train_pct=0.8)

    for df, split in zip([train_df, test_df], ["train", "test"]):
        dataset_path = f"{split}_dataset.jsonl"
        # extract the dataset in jsonl format, save to disk
        dataset = df.apply(format_data_df, axis=1)
        series_to_jsonl(dataset, dataset_path)

        # load the dataset back from disk, to some sanity checks
        dataset = load_dataset(dataset_path)

        print("=" * 80)
        print(f"{split}")
        check_for_errors(dataset)
        token_count_stats(dataset)
        dataset_stats(df)
        print("=" * 80)

        # Upload for finetune purposes to openAI
        upload_finetune_dataset(dataset_path)
