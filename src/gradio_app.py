import copy
import logging
import uuid
from datetime import datetime, timezone

import gradio as gr
import pandas as pd
from buster.busterbot import Buster
from fastapi.encoders import jsonable_encoder

import cfg
from db_utils import Feedback, init_db

mongo_db = init_db()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MAX_TABS = cfg.buster_cfg.retriever_cfg["top_k"]


def get_utc_time() -> str:
    return str(datetime.now(timezone.utc))


def get_session_id() -> str:
    return str(uuid.uuid1())


def check_auth(username: str, password: str) -> bool:
    """Basic auth, only supports a single user."""
    # TODO: update to better auth
    is_auth = username == cfg.USERNAME and password == cfg.PASSWORD
    logger.info(f"Log-in attempted. {is_auth=}")
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


def chat(question, history, document_source, model, user_responses):
    history = history or []

    cfg.buster_cfg.document_source = document_source
    cfg.buster_cfg.completion_cfg["completion_kwargs"]["model"] = model
    buster.update_cfg(cfg.buster_cfg)

    response = buster.process_input(question)

    answer = response.completion.text
    history.append((question, answer))

    sources = format_sources(response.matched_documents)
    sources = pad_sources(sources)

    user_responses.append(response)

    return history, history, user_responses, *sources


def user_responses_formatted(user_responses):
    import json

    responses_copy = copy.deepcopy(user_responses)
    for response in responses_copy:
        # go to json and back to dict so that all int entries are now strings in a dict...
        response.matched_documents = json.loads(
            response.matched_documents.drop(columns=["embedding"]).to_json(orient="index")
        )

    logger.info(responses_copy)

    return responses_copy


def submit_feedback(
    user_responses,
    session_id,
    feedback_good_bad,
    feedback_relevant_length,
    feedback_relevant_answer,
    feedback_relevant_sources,
    feedback_info,
):
    dict_responses = user_responses_formatted(user_responses)
    user_feedback = Feedback(
        good_bad=feedback_good_bad,
        extra_info=feedback_info,
        relevant_answer=feedback_relevant_answer,
        relevant_length=feedback_relevant_length,
        relevant_sources=feedback_relevant_sources,
    )
    feedback = {
        "session_id": session_id,
        "user_responses": dict_responses,
        "feedback": user_feedback,
        "time": get_utc_time(),
    }
    feedback_json = jsonable_encoder(feedback)

    logger.info(feedback_json)
    try:
        # mongo_db["feedback"].replace_one({"_id": session_id}, feedback_json, upsert=True)
        mongo_db["feedback"].insert_one(feedback_json)
        logger.info("response logged to mondogb")
    except Exception as err:
        logger.exception("Something went wrong logging to mongodb")
    # update visibility for extra form
    return {feedback_submitted_message: gr.update(visible=True)}


block = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")

with block:
    buster: Buster = Buster(cfg=cfg.buster_cfg, retriever=cfg.retriever)

    # state variables are client-side and are reset every time a client refreshes the page
    user_responses = gr.State([])
    session_id = gr.State(get_session_id())

    with gr.Row():
        gr.Markdown("<h1><center>LLawMa 🦙: A Question-Answering Bot for your documentation</center></h1>")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### Chatbot")
            chatbot = gr.Chatbot()
            message = gr.Textbox(
                label="Chat with 🦙",
                placeholder="Ask your question here...",
                lines=1,
            )
            submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

            examples = gr.Examples(
                examples=[
                    "What's the state of AI in North America?",
                ],
                inputs=message,
            )
            gr.Markdown("#### Feedback form\nHelp us improve Buster!")
            with gr.Row():
                with gr.Row():
                    feedback_good_bad = gr.Radio(choices=["👍", "👎"], label="How did buster do?")
                    feedback_relevant_length = gr.Radio(
                        choices=["Too Long", "Just Right", "Too Short"], label="How was the answer length?"
                    )
                    feedback_relevant_answer = gr.Slider(
                        minimum=1, maximum=5, label="How relevant was the answer?", interactive=True, value="-", step=1
                    )
                    feedback_relevant_sources = gr.Slider(
                        minimum=1,
                        maximum=5,
                        label="How relevant were the sources?",
                        interactive=True,
                        value="-",
                        step=1,
                    )
                with gr.Row():
                    feedback_info = gr.Textbox(
                        label="Enter additional information (optional)",
                        lines=10,
                        placeholder="Enter more helpful information for us here...",
                    )

                    # feedback_elems = [feedback_good_bad, feedback_relevant_length, feedback_relevant_answer, feedback_relevant_sources, feedback_info]

            submit_feedback_btn = gr.Button("Submit Feedback!")
            with gr.Column(visible=False) as feedback_submitted_message:
                gr.Markdown("Feedback recorded, thank you! 📝")

            submit_feedback_btn.click(
                submit_feedback,
                inputs=[
                    user_responses,
                    session_id,
                    feedback_good_bad,
                    feedback_relevant_length,
                    feedback_relevant_answer,
                    feedback_relevant_sources,
                    feedback_info,
                ],
                outputs=feedback_submitted_message,
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Model")
                # TODO: remove interactive=False flag when deployed model gets access to GPT4
                model = gr.Radio(
                    cfg.available_models, label="Model to use", value=cfg.available_models[0], interactive=False
                )
            with gr.Column(scale=1):
                gr.Markdown("#### Sources")
                case_names = sorted(cfg.document_sources)
                source_dropdown = gr.Dropdown(
                    choices=case_names,
                    value=case_names[0],
                    interactive=True,
                    multiselect=False,
                    label="Source",
                    info="Select a source to query",
                )
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("## References used")
                sources_textboxes = []
                for i in range(MAX_TABS):
                    with gr.Tab(f"Source {i + 1} 📝"):
                        t = gr.Markdown()
                    sources_textboxes.append(t)

    gr.Markdown("This application uses GPT to search the docs for relevant info and answer questions.")

    gr.HTML("️<center> Powered by Buster 🤖")

    state = gr.State()
    agent_state = gr.State()

    submit.click(
        chat,
        inputs=[message, state, source_dropdown, model, user_responses],
        outputs=[chatbot, state, user_responses, *sources_textboxes],
    )
    message.submit(
        chat,
        inputs=[message, state, source_dropdown, model, user_responses],
        outputs=[chatbot, state, user_responses, *sources_textboxes],
    )


# block.launch(debug=True, share=False, auth=check_auth)
block.launch(debug=True, share=False)  # , auth=check_auth)
