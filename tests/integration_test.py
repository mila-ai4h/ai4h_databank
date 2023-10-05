import os

import gradio as gr

import src.cfg as cfg
from src.app_utils import get_logging_db_name, get_session_id, init_db
from src.buster.buster_app import log_completion, submit_feedback
from src.feedback import read_collection

buster = cfg.setup_buster(cfg.buster_cfg)

INSTANCE_TYPE = os.environ["INSTANCE_TYPE"]

# MongoDB Configurations
MONGO_USERNAME = os.environ["MONGO_USERNAME"]
MONGO_PASSWORD = os.environ["MONGO_PASSWORD"]
MONGO_CLUSTER = os.environ["MONGO_CLUSTER"]
MONGO_DATABASE = get_logging_db_name(INSTANCE_TYPE)  # Where all interactions will be stored

db = init_db(username=MONGO_USERNAME, password=MONGO_PASSWORD, cluster=MONGO_CLUSTER, db_name=MONGO_DATABASE)


def test_completion_with_logging():
    """Tests the integration of our capabilities, i.e. running a request, logging it then fetching it back."""

    test_session_id = get_session_id()
    test_username = "unit-test"
    test_user_input = "Has the EU proposed specific legislation on AI?"
    error = True
    num_attempts = 0
    max_attempts = 3
    while error:
        completion = buster.process_input(test_user_input)
        error = completion.error

        num_attempts += 1
        if num_attempts >= max_attempts:
            raise ValueError("Too many tries.")

    assert completion.question_relevant == True
    assert completion.answer_relevant == True

    collection = "interaction"
    log_completion(
        completion=[completion],
        collection=collection,
        request=gr.Request(username=test_username),
        session_id=test_session_id,
        mongo_db=db,
    )

    df = read_collection(mongo_db=db, collection=collection)

    sub_df = df[df.session_id == test_session_id]

    # there should only be one item with the same session ID
    assert len(sub_df) == 1

    record = sub_df.iloc[0]
    assert record["username"] == test_username
    assert record["session_id"] == test_session_id
    # we add a \n after a question so remove it to compare
    assert record["completion_user_input"].strip("\n") == test_user_input