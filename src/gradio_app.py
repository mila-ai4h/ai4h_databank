import logging

import gradio as gr
import pandas as pd
from buster.busterbot import Buster

import cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

buster: Buster = Buster(cfg=cfg.buster_cfg, retriever=cfg.retriever)


def check_auth(username, password):
    """Basic auth, only supports a single user."""
    # TODO: update to better auth
    is_auth = username == cfg.USERNAME and password == cfg.PASSWORD
    logger.info(f"Log-in attempted. {is_auth=}")
    return is_auth


def format_sources(matched_documents: pd.DataFrame):
    formatted_sources = ""

    docs = matched_documents.content.to_list()
    for idx, doc in enumerate(docs):
        formatted_sources += f"### Source {idx + 1} 📝\n" + doc + "\n"

    return formatted_sources


def chat(question, history, document_source):
    history = history or []

    cfg.buster_cfg.document_source = document_source
    buster.update_cfg(cfg.buster_cfg)

    response = buster.process_input(question)

    answer = response.completion.text
    history.append((question, answer))

    sources = format_sources(response.matched_documents)

    return history, history, sources


block = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")

with block:
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

        with gr.Row():
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
                sources_textbox = gr.Markdown()

    gr.Markdown("This application uses GPT to search the docs for relevant info and answer questions.")

    gr.HTML("️<center> Powered by Buster 🤖")

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state, source_dropdown], outputs=[chatbot, state, sources_textbox])
    message.submit(chat, inputs=[message, state, source_dropdown], outputs=[chatbot, state, sources_textbox])


block.launch(debug=True, share=False, auth=check_auth)
