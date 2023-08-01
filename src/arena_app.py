import asyncio
import copy
import logging
import os
import random
from functools import lru_cache
from itertools import zip_longest

import gradio as gr
import pandas as pd
from buster.completers import Completion

import cfg
from app_utils import check_auth
from buster_app import add_sources, get_utc_time
from cfg import buster_cfg, setup_buster
from feedback import ComparisonForm, Feedback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Number of sources to display, get value from cfg
max_sources = cfg.buster_cfg.retriever_cfg["top_k"]

# db that will be logged to
mongo_db = cfg.mongo_db

# Load the sample questions and split them by type
questions = pd.read_csv("sample_questions.csv")
relevant_questions = questions[questions.question_type == "relevant"].question.to_list()

# Set up a version of buster with gpt 3.5
buster_35_cfg = copy.deepcopy(buster_cfg)
buster_35_cfg.completion_cfg["completion_kwargs"]["model"] = "gpt-3.5-turbo"
buster_35 = setup_buster(buster_35_cfg)

# Set up a version of buster with gpt 4
buster_4_cfg = copy.deepcopy(buster_cfg)
buster_4_cfg.completion_cfg["completion_kwargs"]["model"] = "gpt-4"
buster_4 = setup_buster(buster_4_cfg)

# Useful to set when trying out new features locally so you don't have to wait for new streams each time.
# Set to True to enable.
DEBUG_MODE = False


def make_buttons_available():
    enable_btn = gr.Button.update(interactive=True)
    return [enable_btn] * 4


def make_buttons_unavailable():
    disable_btn = gr.Button.update(interactive=False)
    return [disable_btn] * 4


def make_buttons_unfocus():
    unfocus_btn = gr.Button.update(variant="secondary")
    return [unfocus_btn] * 4


def response_recorded_show():
    return gr.Markdown.update(visible=True)


def response_recorded_hide():
    return gr.Markdown.update(visible=False)


def log_submission_leftvote_btn(completion_left, completion_right, current_question, request: gr.Request):
    username = request.username
    vote = "left is better"
    return log_submission(completion_left, completion_right, current_question, vote, username)


def log_submission_rightvote_btn(completion_left, completion_right, current_question, request: gr.Request):
    username = request.username
    vote = "right is better"
    return log_submission(completion_left, completion_right, current_question, vote, username)


def log_submission_tie_btn(completion_left, completion_right, current_question, request: gr.Request):
    username = request.username
    vote = "tied"
    return log_submission(completion_left, completion_right, current_question, vote, username)


def log_submission_bothbad_btn(completion_left, completion_right, current_question, request: gr.Request):
    vote = "both bad"
    username = request.username
    return log_submission(completion_left, completion_right, current_question, vote, username)


def log_submission(completion_left, completion_right, current_question, vote, username):
    model_left = get_model_from_completion(completion_left)
    model_right = get_model_from_completion(completion_right)

    comparison = ComparisonForm(vote=vote, model_left=model_left, model_right=model_right, question=current_question)
    feedback = Feedback(
        username=username,
        user_responses=[completion_left, completion_right],
        feedback_form=comparison,
        time=get_utc_time(),
    )
    feedback.send(mongo_db, collection=cfg.mongo_arena_collection)

    focus_btn = gr.Button.update(variant="primary")
    return focus_btn


async def process_input_35(question):
    """Run buster with chatGPT"""
    return buster_35.process_input(question)


async def process_input_4(question):
    """Run buster with GPT-4"""
    return buster_4.process_input(question)


async def run_models_async(question):
    """Run different buster instances async. Shuffles the resulting models."""
    completion_35, completion_4 = await asyncio.gather(process_input_35(question), process_input_4(question))
    completions = [completion_35, completion_4]
    random.shuffle(completions)
    return completions[0], completions[1]


async def bot_response_stream(
    question: str,
    chatbot_left: gr.Chatbot,
    chatbot_right: gr.Chatbot,
):
    """Given a question, generate the completions."""
    # Clear and init. the conversation
    chatbot_left[-1][1] = ""
    chatbot_right[-1][1] = ""

    if DEBUG_MODE:

        def generator(text):
            for token in text:
                yield token

        completion_left = Completion(
            error=False,
            user_input=question,
            matched_documents=pd.DataFrame(),
            answer_generator=generator("What is Up left"),
            answer_relevant=True,
            question_relevant=True,
            completion_kwargs={"model": "model LEFT"},
        )
        completion_right = Completion(
            error=False,
            user_input=question,
            matched_documents=pd.DataFrame(),
            answer_generator=generator("What is Up right"),
            answer_relevant=True,
            question_relevant=True,
            completion_kwargs={"model": "model RIGHT"},
        )
        completions = [completion_left, completion_right]
        random.shuffle(completions)
        completion_left, completion_right = completions
    else:
        completion_left, completion_right = await run_models_async(question)

    generator_left = completion_left.answer_generator
    generator_right = completion_right.answer_generator

    for token_left, token_right in zip_longest(generator_left, generator_right, fillvalue=""):
        chatbot_left[-1][1] += token_left
        chatbot_right[-1][1] += token_right

        yield chatbot_left, chatbot_right, completion_left, completion_right


def get_model_from_completion(completion: Completion) -> str:
    """Returns the model name of a given completer."""
    return completion.completion_kwargs["model"]


def reveal_models(completion_left: Completion, completion_right: Completion) -> str:
    """Revels the model names as markdown."""
    model_left = get_model_from_completion(completion_left)
    model_right = get_model_from_completion(completion_right)
    return "## Model: " + model_left, "## Model: " + model_right


def hide_models():
    """Resets the model names to empty strings effectively hiding them."""
    return gr.Markdown.update(value=""), gr.Markdown.update(value="")


def update_current_question(textbox, current_question, chatbot_left, chatbot_right):
    """Takes the value from the textbox and sets it as the user's current question state."""
    chatbot_left = [[textbox, None]]
    chatbot_right = [[textbox, None]]

    current_question = copy.copy(textbox)
    textbox = ""
    return textbox, current_question, chatbot_left, chatbot_right


arena_app = gr.Blocks()
with arena_app:
    gr.Markdown("<h1><center>⚔️ Chatbot Arena ⚔️️</center></h1>")

    current_question = gr.State("")
    completion_left = gr.State()
    completion_right = gr.State()

    with gr.Box(elem_id="share-region-anony"):
        with gr.Row():
            with gr.Column():
                model_name_left = gr.Markdown("")
            with gr.Column():
                model_name_right = gr.Markdown("")

        with gr.Row():
            chatbot_left = gr.Chatbot(label="Model A", elem_id=f"chatbot", visible=True, height=550)
            chatbot_right = gr.Chatbot(label="Model B", elem_id=f"chatbot", visible=True, height=550)

        with gr.Box() as button_row:
            with gr.Row():
                leftvote_btn = gr.Button(value="👈  A is better", interactive=False)
                rightvote_btn = gr.Button(value="👉  B is better", interactive=False)
                tie_btn = gr.Button(value="🤝  Tie", interactive=False)
                bothbad_btn = gr.Button(value="👎  Both are bad", interactive=False)
            clr_button = gr.ClearButton(value="Clear 🗑️")
            response_recorded = gr.Markdown("**Succesfully submitted! 💾**", visible=False)

    with gr.Column(variant="panel"):
        gr.Markdown("## References used")
        sources_textboxes = []
        for i in range(3):
            with gr.Tab(f"Source {i + 1} 📝"):
                t = gr.Markdown()
            sources_textboxes.append(t)

    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=True,
                container=False,
                interactive=True,
            )
            gr.Examples(
                examples=relevant_questions,
                inputs=textbox,
                label="Questions users could ask.",
                examples_per_page=50,
            )
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=True)

    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
    ]

    # fmt: off

    # Set up clear button
    clr_button.add(
        components=[
            chatbot_left,
            chatbot_right,
        model_name_left,
        model_name_right,
            response_recorded,
            *sources_textboxes,
        ]
    )
    clr_button.click(
        make_buttons_unavailable,
        outputs=[*btn_list],
    ).then(
        make_buttons_unfocus,
        outputs=[*btn_list],
    )
    # fmt: on

    # fmt: off

    # When clicking the send button, takes care of also making the vote buttons available.
    send_btn.click(
        make_buttons_unfocus,
        outputs=[*btn_list],
    ).then(
        response_recorded_hide,
        outputs=response_recorded,
    ).then(
        update_current_question,
        inputs=[textbox, current_question, chatbot_left, chatbot_right],
        outputs=[textbox, current_question, chatbot_left, chatbot_right],
    ).then(
        hide_models, outputs=[model_name_right, model_name_left]
    ).then(
        bot_response_stream,
        inputs=[current_question, chatbot_left, chatbot_right],
        outputs=[chatbot_left, chatbot_right, completion_left, completion_right],
    ).then(
        add_sources, inputs=[completion_left, gr.State(max_sources)], outputs=[*sources_textboxes]
    ).success(
        make_buttons_available, outputs=[*btn_list]
    )
    # fmt: on


    # fmt: off

    # All the different buttons exhibit the same behaviour on click.
    # 1) register the vote on mongodb, make voted button focused
    # 2) make voting buttons unavailable after vote
    # 3) reveal which model was which
    # 4) show a message that says it was properly logged.
    # If you edit one button, you MUST edit all of them.
    # There might be a more pythonic way to do this in gradio, but I'm not sure exactly how to do this.
    # Can revisit this later


    leftvote_btn.click(
        log_submission_leftvote_btn,
        inputs=[completion_left, completion_right, current_question],
        outputs=leftvote_btn,
    ).then(
        make_buttons_unavailable,
        outputs=[*btn_list],
    ).then(
        reveal_models,
        inputs=[completion_left, completion_right],
        outputs=[model_name_left, model_name_right],
    ).then(
        response_recorded_show,
        outputs=response_recorded,
    )

    rightvote_btn.click(
        log_submission_rightvote_btn,
        inputs=[completion_left, completion_right, current_question],
        outputs=rightvote_btn,
    ).then(
        make_buttons_unavailable,
        outputs=[*btn_list],
    ).then(
        reveal_models,
        inputs=[completion_left, completion_right],
        outputs=[model_name_left, model_name_right],
    ).then(
        response_recorded_show,
        outputs=response_recorded,
    )

    tie_btn.click(
        log_submission_tie_btn,
        inputs=[completion_left, completion_right, current_question],
        outputs=tie_btn,
    ).then(
        make_buttons_unavailable,
        outputs=[*btn_list],
    ).then(
        reveal_models,
        inputs=[completion_left, completion_right],
        outputs=[model_name_left, model_name_right],
    ).then(
        response_recorded_show,
        outputs=response_recorded,
    )

    bothbad_btn.click(
        log_submission_bothbad_btn,
        inputs=[completion_left, completion_right, current_question],
        outputs=bothbad_btn,
    ).then(
        make_buttons_unavailable,
        outputs=[*btn_list],
    ).then(
        reveal_models,
        inputs=[completion_left, completion_right],
        outputs=[model_name_left, model_name_right],
    ).then(
        response_recorded_show,
        outputs=response_recorded,
    )
    # fmt: on


if __name__ == "arena_app":
    arena_app.queue(concurrency_count=16)
    arena_app.launch(share=False, auth=check_auth)

else:
    arena_app.auth = check_auth
    arena_app.auth_message = ""
    arena_app.queue()
