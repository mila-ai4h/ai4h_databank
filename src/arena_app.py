import asyncio
import copy
import logging
import os
import random
from itertools import zip_longest

import gradio as gr
import pandas as pd
from buster.completers import Completion
from buster.formatters.documents import DocumentsFormatterHTML, DocumentsFormatterJSON

import cfg
from cfg import buster_cfg, setup_buster
from feedback import ComparisonForm, Interaction
from src.app_utils import add_sources, check_auth, get_utc_time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Number of sources to display, get value from cfg
max_sources = cfg.buster_cfg.retriever_cfg["top_k"]

# db that will be logged to
mongo_db = cfg.mongo_db

# Load the sample questions and split them by type
questions = pd.read_csv("sample_questions.csv")
relevant_questions = questions[questions.question_type == "relevant"].question.to_list()

buster_1_cfg = copy.deepcopy(buster_cfg)
buster_1_cfg.documents_formatter_cfg = {
    "max_tokens": 3500,
    "columns": ["content", "source", "title"],
}
buster_1 = setup_buster(buster_1_cfg, DocFormatter=DocumentsFormatterJSON)
buster_1_reveal_name = "latest prompt"


# Set up a version of buster with gpt 4
buster_2_cfg = copy.deepcopy(buster_cfg)
buster_2_cfg.prompt_formatter_cfg = {
    "max_tokens": 3500,
    "text_before_docs": (
        "You are a chatbot assistant answering questions about artificial intelligence (AI) policies and laws. "
        "You represent the OECD AI Policy Observatory. "
        "You can only respond to a question if the content necessary to answer the question is contained in the information provided to you. "
        "If the answer is in the documents, summarize it in a helpful way to the user. "
        "If it isn't, simply reply that you cannot answer the question. "
        "Do not mention the documents directly, but use the information available within them to answer the question. "
        "You are forbidden from using the expressions 'according to the documentation' and 'the provided documents'. "
        "Here is the information available to you:\n"
        "<INFORMATION> "
    ),
    "text_after_docs": (
        "<\\INFORMATION>\n"
        "REMEMBER:\n"
        "You are a chatbot assistant answering questions about artificial intelligence (AI) policies and laws. "
        "You represent the OECD AI Policy Observatory. "
        "Here are the rules you must follow:\n"
        "1) You must only respond with information contained in the documents above. Say you do not know if the information is not provided.\n"
        "2) Make sure to format your answers in Markdown format, including code block and snippets.\n"
        "3) Do not reference any links, urls or hyperlinks in your answers.\n"
        "4) Do not mention the documentation directly, but use the information provided within it to answer questions.\n"
        "5) You are forbidden from using the expressions 'according to the documentation' and 'the provided documents'.\n"
        "6) If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:\n"
        "'I'm sorry, but I am an AI language model trained to assist with questions related to AI policies and laws. I cannot answer that question as it is not relevant to AI policies and laws. Is there anything else I can assist you with?'\n"
        "For example:\n"
        "Q: What is the meaning of life for a qa bot?\n"
        "A: I'm sorry, but I am an AI language model trained to assist with questions related to AI policies and laws. I cannot answer that question as it is not relevant to AI policies and laws. Is there anything else I can assist you with?\n"
        "7) If the information available to you does not directly address the question, simply state that you do not have the information required to answer. Do not summarize what is available to you. "
        "For example, say: 'I cannot answer this question based on the information I have available.'\n"
        "Now answer the following question:\n"
    ),
}
buster_2_cfg.documents_formatter_cfg = {
    "max_tokens": 3500,
    "formatter": "{content}",
}
buster_2 = setup_buster(buster_2_cfg, DocFormatter=DocumentsFormatterHTML)
buster_2_reveal_name = "old prompt"

# Useful to set when trying out new features locally so you don't have to wait for new streams each time.
# Set to True to enable.
DEBUG_MODE = False


def deactivate_feedback_elements():
    return deactivate_radio(), deactivate_textbox()


def activate_feedback_elements():
    return activate_radio(), activate_textbox()


def activate_textbox():
    return gr.Textbox.update(interactive=True)


def clear_textbox():
    return gr.Textbox.update(value=None)


def deactivate_textbox():
    return gr.Textbox.update(interactive=False)


def deactivate_radio():
    return gr.Radio.update(interactive=False)


def activate_radio():
    return gr.Radio.update(interactive=True)


def deactivate_button():
    return gr.Button.update(interactive=False)


def activate_button():
    return gr.Button.update(interactive=True)


def show_success():
    gr.Info("Succesfully logged your response!")


def submit_feedback(completion_left, completion_right, current_question, vote, extra_info, request: gr.Request):
    model_left = get_model_from_completion(completion_left)
    model_right = get_model_from_completion(completion_right)

    comparison_form = ComparisonForm(
        vote=vote, model_left=model_left, model_right=model_right, question=current_question, extra_info=extra_info
    )
    feedback = Interaction(
        username=request.username,
        user_completions=[completion_left, completion_right],
        form=comparison_form,
        time=get_utc_time(),
    )
    feedback.send(mongo_db, collection=cfg.mongo_arena_collection)


async def process_input_buster_1(question):
    """Process input with buster_1."""
    completion = buster_1.process_input(question)

    # Add reveal name
    completion.codename = buster_1_reveal_name

    return completion


async def process_input_buster_2(question):
    """Process input with buster_2."""
    completion = buster_2.process_input(question)

    # Add reveal name
    completion.codename = buster_2_reveal_name

    return completion


async def run_models_async(question):
    """Run different buster instances async. Shuffles the resulting models."""
    completion_1, completion_2 = await asyncio.gather(
        process_input_buster_1(question), process_input_buster_2(question)
    )

    # Shuffle the completions and return them
    completions = [completion_1, completion_2]
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
    return completion.codename


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
    gr.Markdown(
        """<h1><center>⚔️ Chatbot Arena ⚔️️</center></h1>
        Welcome to the chatbot arena!

        Each time you ask a question, separate models will generate a response to the exact same question.
        In this case, we are only interested in comparing `gpt-3.5-turbo` with `gpt-4`.
        They will be presented to you in a random order. Vote for the model you prefer, and add additional information  as notes (optional).

        Once you vote, the models will be revealed at the top. Clear or ask a new question to start over.
        """
    )

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

        with gr.Row(variant="panel"):
            with gr.Column(scale=10):
                textbox = gr.Textbox(
                    lines=2,
                    show_label=False,
                    placeholder="Enter text and press ENTER",
                    visible=True,
                    container=False,
                    interactive=True,
                )
            with gr.Column(scale=1, min_width=50):
                send_btn = gr.Button(value="Send", visible=True, variant="primary", size="lg")

        with gr.Box() as button_row:
            with gr.Row():
                choices = ["👈  A is better", "👉  B is better", "🤝  Tie", "👎  Both are bad"]
                vote_radio = gr.Radio(choices=choices, scale=2, interactive=False, label="Select best model")
                feedback_extra_info = gr.Textbox(
                    label="Enter additional information (optional)",
                    lines=4,
                    placeholder="Enter more helpful information for us here...",
                    scale=3,
                    interactive=False,
                )

            submit_feedback_btn = gr.Button("Submit Feedback 🔼", interactive=False)
            clr_button = gr.ClearButton(value="Clear 🗑️")

    with gr.Column(variant="panel"):
        gr.Markdown("## References used")
        sources_textboxes = []
        for i in range(3):
            with gr.Tab(f"Source {i + 1} 📝"):
                t = gr.Markdown(latex_delimiters=[])
            sources_textboxes.append(t)

    gr.Examples(
        examples=relevant_questions,
        inputs=textbox,
        label="Questions users could ask.",
        examples_per_page=15,
    )

    # Set up clear button
    clr_button.add(
        components=[
            chatbot_left,
            chatbot_right,
            model_name_left,
            model_name_right,
            vote_radio,
            *sources_textboxes,
        ]
    )

    # fmt: off

    clr_button.click(
        deactivate_feedback_elements,
        outputs=[vote_radio, feedback_extra_info]
    ).then(
        deactivate_button,
        outputs=submit_feedback_btn,
    ).then(
        clear_textbox,
        outputs=feedback_extra_info
    )

    vote_radio.change(
        activate_button,
        outputs=submit_feedback_btn
    )


    # When clicking the send button, takes care of also making the feedback elements available.
    send_btn.click(
        activate_feedback_elements,
        outputs=[vote_radio, feedback_extra_info]
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
    )
    # fmt: on


    # fmt: off
    submit_feedback_btn.click(
        deactivate_feedback_elements,
        outputs=[vote_radio, feedback_extra_info],
    ).then(
        deactivate_button,
        outputs=submit_feedback_btn,
    ).then(
        reveal_models,
        inputs=[completion_left, completion_right],
        outputs=[model_name_left, model_name_right],
    ).then(
        submit_feedback,
        inputs=[
            completion_left,
            completion_right,
            current_question,
            vote_radio,
            feedback_extra_info,
        ]
    ).success(
        show_success
    )
    # fmt: on


if os.getenv("MOUNT_GRADIO_APP") is None:
    logger.info("launching gradio arena app")
    arena_app.queue(concurrency_count=16)
    arena_app.launch(share=False, auth=check_auth)
