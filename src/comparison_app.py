import asyncio
import copy
import logging
import os
import random
from functools import lru_cache
from itertools import zip_longest

import gradio as gr
import pandas as pd

import cfg
from app_utils import check_auth
from buster_app import add_sources, get_utc_time
from cfg import buster_cfg, setup_buster
from feedback import ComparisonForm, Feedback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

max_sources = cfg.buster_cfg.retriever_cfg["top_k"]

mongo_db = cfg.mongo_db

# Load the sample questions and split them by type
questions = pd.read_csv("sample_questions.csv")
relevant_questions = questions[questions.question_type == "relevant"].question.to_list()

enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)
focus_btn = gr.Button.update(variant="primary")
unfocus_btn = gr.Button.update(variant="secondary")

num_sides = 2
models = []

buster_35_cfg = copy.deepcopy(buster_cfg)
buster_35_cfg.completion_cfg["completion_kwargs"]["model"] = "gpt-3.5-turbo"
buster_35 = setup_buster(buster_35_cfg)

buster_4_cfg = copy.deepcopy(buster_cfg)
buster_4_cfg.completion_cfg["completion_kwargs"]["model"] = "gpt-4"
buster_4 = setup_buster(buster_4_cfg)


# Define your asynchronous functions
async def process_input_35(question):
    return buster_35.process_input(question)


async def process_input_4(question):
    return buster_4.process_input(question)


# Create a separate async function that uses asyncio.gather()
async def run_async_tasks(question):
    completion_35, completion_4 = await asyncio.gather(process_input_35(question), process_input_4(question))
    completors = [completion_35, completion_4]
    random.shuffle(completors)
    return completors[0], completors[1]


async def bot_response_stream(
    question,
    chatbot_left,
    chatbot_right,
):
    # Init the conversation
    chatbot_left[-1][1] = ""
    chatbot_right[-1][1] = ""

    def generator(text):
        for token in text:
            yield token

    debug = False
    if debug:
        from buster.completers import Completion

        completor_left = Completion(
            error=False,
            user_input=question,
            matched_documents=pd.DataFrame(),
            answer_generator=generator("What is Up left"),
            answer_relevant=True,
            question_relevant=True,
            completion_kwargs={"model": "model LEFT"},
        )
        completor_right = Completion(
            error=False,
            user_input=question,
            matched_documents=pd.DataFrame(),
            answer_generator=generator("What is Up right"),
            answer_relevant=True,
            question_relevant=True,
            completion_kwargs={"model": "model RIGHT"},
        )
        completors = [completor_left, completor_right]
        random.shuffle(completors)
        completor_left, completor_right = completors
    else:
        completor_left, completor_right = await run_async_tasks(question)

    generator_left = completor_left.answer_generator
    generator_right = completor_right.answer_generator

    for token_left, token_right in zip_longest(generator_left, generator_right, fillvalue=""):
        chatbot_left[-1][1] += token_left
        chatbot_right[-1][1] += token_right

        yield chatbot_left, chatbot_right, completor_left, completor_right


def get_model_from_completor(completor):
    return completor.completion_kwargs["model"]


def reveal_models(completor_left, completor_right):
    model_left = get_model_from_completor(completor_left)
    model_right = get_model_from_completor(completor_right)
    return "## Model: " + model_left, "## Model: " + model_right


def hide_models():
    return gr.Markdown.update(value=""), gr.Markdown.update(value="")


def update_current_question(textbox, current_question, chatbot_left, chatbot_right):
    chatbot_left = [[textbox, None]]
    chatbot_right = [[textbox, None]]

    current_question = copy.copy(textbox)
    textbox = ""
    return textbox, current_question, chatbot_left, chatbot_right


comparison_app = gr.Blocks()
with comparison_app:
    gr.Markdown("<h1><center>⚔️ Chatbot Arena ⚔️️</center></h1>")

    current_question = gr.State("")
    completor_left = gr.State()
    completor_right = gr.State()

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

    def make_buttons_available():
        return [enable_btn] * 4

    def make_buttons_unavailable():
        return [disable_btn] * 4

    def make_buttons_unfocus():
        return [unfocus_btn] * 4

    def response_recorded_show():
        return gr.Markdown.update(visible=True)

    def response_recorded_hide():
        return gr.Markdown.update(visible=False)

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
    ]

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
        outputs=[chatbot_left, chatbot_right, completor_left, completor_right],
    ).then(
        add_sources, inputs=[completor_left, gr.State(max_sources)], outputs=[*sources_textboxes]
    ).success(
        make_buttons_available, outputs=[*btn_list]
    )

    def log_submission_leftvote_btn(completor_left, completor_right, current_question, request: gr.Request):
        username = request.username
        vote = "left is better"
        return log_submission(completor_left, completor_right, current_question, vote, username)

    def log_submission_rightvote_btn(completor_left, completor_right, current_question, request: gr.Request):
        username = request.username
        vote = "right is better"
        return log_submission(completor_left, completor_right, current_question, vote, username)

    def log_submission_tie_btn(completor_left, completor_right, current_question, request: gr.Request):
        username = request.username
        vote = "tied"
        return log_submission(completor_left, completor_right, current_question, vote, username)

    def log_submission_bothbad_btn(completor_left, completor_right, current_question, request: gr.Request):
        vote = "both bad"
        username = request.username
        return log_submission(completor_left, completor_right, current_question, vote, username)

    def log_submission(completor_left, completor_right, current_question, vote, username):
        model_left = get_model_from_completor(completor_left)
        model_right = get_model_from_completor(completor_right)

        comparison = ComparisonForm(
            vote=vote, model_left=model_left, model_right=model_right, question=current_question
        )
        feedback = Feedback(
            username=username,
            user_responses=[completor_left, completor_right],
            feedback_form=comparison,
            time=get_utc_time(),
        )
        feedback.send(mongo_db, collection=cfg.mongo_arena_collection)
        return focus_btn

    leftvote_btn.click(
        log_submission_leftvote_btn,
        inputs=[completor_left, completor_right, current_question],
        outputs=leftvote_btn,
    ).then(
        make_buttons_unavailable,
        outputs=[*btn_list],
    ).then(
        reveal_models,
        inputs=[completor_left, completor_right],
        outputs=[model_name_left, model_name_right],
    ).then(
        response_recorded_show,
        outputs=response_recorded,
    )

    rightvote_btn.click(
        log_submission_rightvote_btn,
        inputs=[completor_left, completor_right, current_question],
        outputs=rightvote_btn,
    ).then(
        make_buttons_unavailable,
        outputs=[*btn_list],
    ).then(
        reveal_models,
        inputs=[completor_left, completor_right],
        outputs=[model_name_left, model_name_right],
    ).then(
        response_recorded_show,
        outputs=response_recorded,
    )

    tie_btn.click(
        log_submission_tie_btn,
        inputs=[completor_left, completor_right, current_question],
        outputs=tie_btn,
    ).then(
        make_buttons_unavailable,
        outputs=[*btn_list],
    ).then(
        reveal_models,
        inputs=[completor_left, completor_right],
        outputs=[model_name_left, model_name_right],
    ).then(
        response_recorded_show,
        outputs=response_recorded,
    )

    bothbad_btn.click(
        log_submission_bothbad_btn,
        inputs=[completor_left, completor_right, current_question],
        outputs=bothbad_btn,
    ).then(
        make_buttons_unavailable,
        outputs=[*btn_list],
    ).then(
        reveal_models,
        inputs=[completor_left, completor_right],
        outputs=[model_name_left, model_name_right],
    ).then(
        response_recorded_show,
        outputs=response_recorded,
    )


# comparison_app.auth = check_auth
# comparison_app.auth_message = ""
# comparison_app.queue
comparison_app.queue(concurrency_count=16)
comparison_app.launch(share=False, auth=check_auth)()
