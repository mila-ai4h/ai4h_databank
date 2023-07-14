from functools import lru_cache
import os
import cfg
import logging
import gradio as gr
import pandas as pd
from copy import copy

from db_utils import init_db
from feedback import Feedback, FeedbackForm
from buster_app import check_auth

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

username = os.getenv("AI4H_MONGODB_USERNAME")
password = os.getenv("AI4H_MONGODB_PASSWORD")
cluster = os.getenv("AI4H_MONGODB_CLUSTER")
db_name = os.getenv("AI4H_MONGODB_DB_NAME")
mongo_db = init_db(username, password, cluster, db_name)

retriever = cfg.retriever


# Load the sample questions and split them by type
questions = pd.read_csv("sample_questions.csv")
relevant_questions = questions[questions.question_type == "relevant"].question.to_list()

annotation_app = gr.Blocks()

top_k = 10
empty_documents = [""] * top_k
empty_evaluations = [False] * top_k


@lru_cache
def retrieve(query, top_k):
    return retriever.retrieve(query=query, top_k=top_k)


def get_relevant_documents(
    query: str,
    current_question,
    top_k=10,
) -> list[str]:
    current_question = copy(query)

    if query == "":
        return empty_documents

    matched_documents = retrieve(query=query, top_k=top_k)
    return matched_documents.content.to_list(), current_question


def save_to_db(user_evaluations, request: gr.Request):
    collection = "chunk_annotation"

    username: str = request.username

    eval = {
        "username": username,
        "evaluations": user_evaluations,
    }
    mongo_db[collection].replace_one({"username": username}, eval, upsert=True)


def get_document(idx, documents):
    return documents[idx]


with annotation_app:
    # TODO: trigger a proper change to update

    # state variables are client-side and are reset every time a client refreshes the page
    user_responses = gr.State([])

    # keep track of submission form components here...
    documents = gr.State(empty_documents)
    user_evaluations = gr.State({})

    # current_question is different from question_input because the question_input can be changed after asking a question
    current_question = gr.State("")

    gr.Markdown("<h1><center>Reference Annotation</center></h1>")

    with gr.Column(scale=2):
        question_input = gr.Textbox(
            label="Ask your question",
            placeholder="Ask your question here...",
            lines=1,
        )
    with gr.Column(variant="panel"):
        gr.Markdown("## Example questions")
        with gr.Tab("Relevant questions"):
            gr.Examples(
                examples=relevant_questions,
                inputs=question_input,
                label="Questions users could ask.",
                examples_per_page=50,
            )

    def update_documents(documents):
        updated_document_content = []

        for doc in documents:
            updated_document_content.append(doc)

        return updated_document_content

    document_evaluation = []
    document_content = []

    with gr.Row():
        with gr.Column():
            with gr.Row():
                ask_button = gr.Button(value="Ask", variant="primary")
                save_button = gr.Button(value="Save 💾", variant="primary")
            for idx in range(len(documents.value)):
                with gr.Column():
                    document_evaluation.append(gr.Checkbox(value=False, label="relevant", interactive=True))
                    document_content.append(
                        gr.Textbox(label=f"Document", interactive=False, value=documents.value[idx])
                    )

    def update_evaluations(question: str, user_evaluations):
        latest_evaluation = user_evaluations.get(question)
        if latest_evaluation is None:
            # return an empty evaluation
            return empty_evaluations
        else:
            return latest_evaluation

    def save_evaluations(question, user_evaluations, *current_evaluations):
        user_evaluations[question] = list(current_evaluations)
        return user_evaluations

    def clear_documents():
        """Simulates a loading strategy"""
        return ["loading..."] * top_k

    def clear_evaluations():
        """Simulates a loading strategy"""
        return empty_evaluations

    # fmt: off
    ask_button.click(
        fn=clear_documents, outputs=document_content,
    ).then(
        fn=clear_evaluations, outputs=document_evaluation,
    ).then(
        fn=get_relevant_documents, inputs=[question_input, current_question], outputs=[documents, current_question]
    ).then(
        update_documents, inputs=[documents], outputs=document_content
    ).then(
        update_evaluations, inputs=[current_question, user_evaluations], outputs=document_evaluation
    )

    save_button.click(
        fn=save_evaluations, inputs=[current_question, user_evaluations, *document_evaluation], outputs=user_evaluations
    ).then(
        fn=save_to_db, inputs=user_evaluations
    )
    # fmt: on

annotation_app.auth = check_auth
annotation_app.auth_message = ""
