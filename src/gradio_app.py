import logging
import os
from datetime import datetime, timezone

import gradio as gr
import pandas as pd
from buster.busterbot import Buster

import cfg
from db_utils import init_db
from feedback import Feedback, FeedbackForm

username = os.getenv("AI4H_MONGODB_USERNAME")
password = os.getenv("AI4H_MONGODB_PASSWORD")
cluster = os.getenv("AI4H_MONGODB_CLUSTER")
db_name = os.getenv("AI4H_MONGODB_DB_NAME")
mongo_db = init_db(username, password, cluster, db_name)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

buster = Buster(retriever=cfg.retriever, completer=cfg.completer, validator=cfg.validator)

MAX_TABS = cfg.buster_cfg.retriever_cfg["top_k"]
RELEVANT_QUESTIONS = pd.read_csv("Questions dataset - Relevant.csv", header=None)[0].to_list()
IRRELEVANT_QUESTIONS = pd.read_csv("Questions dataset - Irrelevant.csv", header=None)[0].to_list()
TRICK_QUESTIONS = pd.read_csv("Questions dataset - Trick.csv", header=None)[0].to_list()


def get_utc_time() -> str:
    return str(datetime.now(timezone.utc))


def check_auth(username: str, password: str) -> bool:
    """Check if authentication succeeds or not.

    The authentication leverages the built-in gradio authentication. We use a shared password among users.
    It is temporary for developing the PoC. Proper authentication needs to be implemented in the future.
    We allow a valid username to be any username beginning with 'databank-', this will allow us to differentiate between users easily.
    """
    valid_user = username.startswith("databank-") or username == cfg.USERNAME
    valid_password = password == cfg.PASSWORD
    is_auth = valid_user and valid_password
    logger.info(f"Log-in attempted by {username=}. {is_auth=}")
    return is_auth


def format_sources(matched_documents: pd.DataFrame) -> list[str]:
    formatted_sources = []

    for _, doc in matched_documents.iterrows():
        formatted_sources.append(f"### [{doc.title}]({doc.url})\n{doc.content}\n")

    return formatted_sources


def pad_sources(sources: list[str]) -> list[str]:
    """Pad sources with empty strings to ensure that the number of tabs is always MAX_TABS."""
    k = len(sources)
    return sources + [""] * (MAX_TABS - k)


def add_sources(completion):
    completion = buster.postprocess_completion(completion)

    if any(arg is False for arg in [completion.question_relevant, completion.answer_relevant]):
        # Question was not relevant, don't bother doing anything else...
        formatted_sources = [""]
        formatted_sources = pad_sources(formatted_sources)
        return formatted_sources

    formatted_sources = format_sources(completion.matched_documents)

    return formatted_sources


def append_response(response, user_responses):
    user_responses.append(response)
    return user_responses


def user(user_input, history):
    """Adds user's question immediately to the chat."""
    return "", history + [[user_input, None]]


def chat(history):
    user_input = history[-1][0]

    response = buster.process_input(user_input)

    history[-1][1] = ""

    for token in response.completor:
        history[-1][1] += token

        yield history, response


def submit_feedback(
    user_responses,
    feedback_relevant_sources,
    feedback_relevant_answer,
    feedback_info,
    request: gr.Request,
):
    feedback_form = FeedbackForm(
        extra_info=feedback_info,
        relevant_answer=feedback_relevant_answer,
        relevant_sources=feedback_relevant_sources,
    )
    feedback = Feedback(
        user_responses=user_responses,
        feedback_form=feedback_form,
        time=get_utc_time(),
        username=request.username,
    )
    feedback.send(mongo_db, collection=cfg.feedback_collection)


def toggle_feedback_visible(visible: bool):
    return {feedback_submitted_message: gr.update(visible=visible)}


block = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")

with block:
    # TODO: trigger a proper change to update

    # state variables are client-side and are reset every time a client refreshes the page
    user_responses = gr.State([])

    with gr.Row():
        gr.Markdown("<h1><center>LLawMa 🦙: A Question-Answering Bot for your documentation</center></h1>")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Chatbot")
            chatbot = gr.Chatbot()
            message = gr.Textbox(
                label="Chat with 🦙",
                placeholder="Ask your question here...",
                lines=1,
            )
            submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

            with gr.Column(variant="panel"):
                gr.Markdown("## Example questions")
                with gr.Tab("Relevant questions"):
                    gr.Examples(
                        examples=RELEVANT_QUESTIONS,
                        inputs=message,
                        label="Questions users could ask.",
                    )
                with gr.Tab("Irrelevant questions"):
                    gr.Examples(
                        examples=IRRELEVANT_QUESTIONS,
                        inputs=message,
                        label="Questions with no relevance to the OECD AI Policy Observatory.",
                    )
                with gr.Tab("Trick questions"):
                    gr.Examples(
                        examples=TRICK_QUESTIONS,
                        inputs=message,
                        label="Questions about non-existing AI policies and laws.",
                    )

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Model")
                # TODO: remove interactive=False flag when deployed model gets access to GPT4
                model = gr.Radio(
                    cfg.available_models, label="Model to use", value=cfg.available_models[0], interactive=False
                )
            with gr.Column(variant="panel"):
                gr.Markdown("## References used")
                sources_textboxes = []
                for i in range(MAX_TABS):
                    with gr.Tab(f"Source {i + 1} 📝"):
                        t = gr.Markdown()
                    sources_textboxes.append(t)

    # Feedback
    with gr.Column(variant="panel"):
        gr.Markdown("## Feedback form\nHelp us improve LLawMa 🦙!")
        with gr.Row():
            feedback_relevant_sources = gr.Radio(
                choices=["👍", "👎"], label="Were any of the retrieved sources relevant?"
            )
        with gr.Row():
            feedback_relevant_answer = gr.Radio(choices=["👍", "👎"], label="Was the generated answer useful?")

        feedback_info = gr.Textbox(
            label="Enter additional information (optional)",
            lines=10,
            placeholder="Enter more helpful information for us here...",
        )

        submit_feedback_btn = gr.Button("Submit Feedback!")
        with gr.Column(visible=False) as feedback_submitted_message:
            gr.Markdown("Feedback recorded, thank you! 📝")

    # fmt: off
    submit_feedback_btn.click(
        submit_feedback,
        inputs=[
            user_responses,
            feedback_relevant_sources,
            feedback_relevant_answer,
            feedback_info,
        ],
    ).then(
        toggle_feedback_visible,
        inputs=gr.State(False),
        outputs=feedback_submitted_message,
    ).success(
        toggle_feedback_visible,
        inputs=gr.State(True),
        outputs=feedback_submitted_message,
    )
    # fmt: on

    gr.Markdown("This application uses GPT to search the docs for relevant info and answer questions.")

    gr.HTML("<center> Powered by <a href='https://github.com/jerpint/buster'>Buster</a> 🤖</center>")

    response = gr.State()

    # fmt: off
    submit.click(
        user, [message, chatbot], [message, chatbot]
    ).then(
        toggle_feedback_visible,
        inputs=gr.State(False),
        outputs=feedback_submitted_message,
    ).then(
        chat,
        inputs=[chatbot],
        outputs=[chatbot, response],
    ).then(
        add_sources,
        inputs=[response],
        outputs=[*sources_textboxes]
    ).then(
        append_response,
        inputs=[response, user_responses], outputs=[user_responses]
    )
    message.submit(
        user, [message, chatbot], [message, chatbot]
    ).then(
        toggle_feedback_visible,
        inputs=gr.State(False),
        outputs=feedback_submitted_message,
    ).then(
        chat,
        inputs=[chatbot],
        outputs=[chatbot, response],
    ).then(
        add_sources,
        inputs=[response],
        outputs=[*sources_textboxes]
    ).then(
        append_response,
        inputs=[response, user_responses], outputs=[user_responses]
    )
    # fmt: on

block.queue(concurrency_count=16)
block.launch(share=False, auth=check_auth)
