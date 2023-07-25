import copy
import logging
import os

import gradio as gr
import pandas as pd

import cfg
from app_utils import add_sources, check_auth, get_utc_time
from cfg import setup_buster
from feedback import Feedback, FeedbackForm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mongo_db = cfg.mongo_db
buster_cfg = copy.deepcopy(cfg.buster_cfg)
buster = setup_buster(buster_cfg=buster_cfg)
max_sources = cfg.buster_cfg.retriever_cfg["top_k"]

# Load the sample questions and split them by type
questions = pd.read_csv("sample_questions.csv")
relevant_questions = questions[questions.question_type == "relevant"].question.to_list()
irrelevant_questions = questions[questions.question_type == "irrelevant"].question.to_list()
trick_questions = questions[questions.question_type == "trick"].question.to_list()


def append_response(response, user_responses):
    user_responses.append(response)
    return user_responses


def user(user_input, history):
    """Adds user's question immediately to the chat."""
    return "", history + [[user_input, None]]


def chat(history):
    user_input = history[-1][0]

    completion = buster.process_input(user_input)

    history[-1][1] = ""

    for token in completion.answer_generator:
        history[-1][1] += token

        yield history, completion


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
    feedback.send(mongo_db, collection=cfg.mongo_feedback_collection)


def toggle_feedback_visible(visible: bool):
    """Toggles the visibility of the 'feedback submitted' message."""
    return {feedback_submitted_message: gr.update(visible=visible)}


def clear_feedback_form():
    """Clears the contents of the feedback form."""
    return {
        feedback_submitted_message: gr.update(visible=False),
        feedback_relevant_sources: gr.update(value=None),
        feedback_relevant_answer: gr.update(value=None),
        feedback_info: gr.update(value=""),
    }


buster_app = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")

with buster_app:
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
            submit = gr.Button(value="Send", variant="primary")

            with gr.Column(variant="panel"):
                gr.Markdown("## Example questions")
                with gr.Tab("Relevant questions"):
                    gr.Examples(
                        examples=relevant_questions,
                        inputs=message,
                        label="Questions users could ask.",
                    )
                with gr.Tab("Irrelevant questions"):
                    gr.Examples(
                        examples=irrelevant_questions,
                        inputs=message,
                        label="Questions with no relevance to the OECD AI Policy Observatory.",
                    )
                with gr.Tab("Trick questions"):
                    gr.Examples(
                        examples=trick_questions,
                        inputs=message,
                        label="Questions about non-existing AI policies and laws.",
                    )

        with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("## References used")
                sources_textboxes = []
                for i in range(max_sources):
                    with gr.Tab(f"Source {i + 1} 📝"):
                        t = gr.Markdown()
                    sources_textboxes.append(t)

            with gr.Column():
                gr.Markdown("## Parameters")
                metadata = [
                    ("generation model", cfg.buster_cfg.completion_cfg["completion_kwargs"]["model"]),
                    ("embedding model", cfg.buster_cfg.retriever_cfg["embedding_model"]),
                ]
                gr.HighlightedText(value=metadata, label="Parameters")

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
        toggle_feedback_visible,
        inputs=gr.State(False),
        outputs=feedback_submitted_message,
    ).then(
        submit_feedback,
        inputs=[
            user_responses,
            feedback_relevant_sources,
            feedback_relevant_answer,
            feedback_info,
        ],
    ).success(
        toggle_feedback_visible,
        inputs=gr.State(True),
        outputs=feedback_submitted_message,
    )
    # If you rage click the subimt feedback button, it re-appears so you are confident it was recorded properly.
    # fmt: on

    gr.Markdown("This application uses GPT to search the docs for relevant info and answer questions.")

    gr.HTML("<center> Powered by <a href='https://github.com/jerpint/buster'>Buster</a> 🤖</center>")

    response = gr.State()

    # fmt: off
    submit.click(
        user, [message, chatbot], [message, chatbot]
    ).then(
        clear_feedback_form,
        outputs=[feedback_submitted_message, feedback_relevant_sources, feedback_relevant_answer, feedback_info]
    ).then(
        chat,
        inputs=[chatbot],
        outputs=[chatbot, response],
    ).then(
        add_sources,
        inputs=[response, gr.State(max_sources)],
        outputs=[*sources_textboxes]
    ).then(
        append_response,
        inputs=[response, user_responses], outputs=[user_responses]
    )
    message.submit(
        user, [message, chatbot], [message, chatbot]
    ).then(
        clear_feedback_form,
        outputs=[feedback_submitted_message, feedback_relevant_sources, feedback_relevant_answer, feedback_info]
    ).then(
        chat,
        inputs=[chatbot],
        outputs=[chatbot, response],
    ).then(
        add_sources,
        inputs=[response, gr.State(max_sources)],
        outputs=[*sources_textboxes]
    ).then(
        append_response,
        inputs=[response, user_responses], outputs=[user_responses]
    )
    # fmt: on

if __name__ == "__main__":
    buster_app.queue(concurrency_count=16)
    buster_app.launch(share=False, auth=check_auth)

else:
    buster_app.auth = check_auth
    buster_app.auth_message = ""
    buster_app.queue()
