import os
import logging

import gradio as gr
import openai
from huggingface_hub import hf_hub_download

from buster.busterbot import Buster
from buster.retriever import Retriever
from buster.utils import get_retriever_from_extension
import cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# initialize buster with the config in config.py (adapt to your needs) ...
retriever: Retriever = get_retriever_from_extension(cfg.documents_filepath)(
    cfg.documents_filepath
)
buster: Buster = Buster(cfg=cfg.buster_cfg, retriever=retriever)

# auth information
USERNAME = os.getenv("AI4H_USERNAME")
PASSWORD = os.getenv("AI4H_PASSWORD")

# hf hub information
REPO_ID = "jerpint/databank-ai4h"
DB_FILE = "documents.db"
HUB_TOKEN = os.environ.get("HUB_TOKEN")

# set openAI creds
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")


# download the documents.db hosted on the dataset space
logger.info(f"Downloading {DB_FILE} from hub...")
hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=DB_FILE,
    token=HUB_TOKEN,
    local_dir=".",
)
logger.info(f"Downloaded.")


def check_auth(username, password):
    """Basic auth, only supports a single user."""
    # TODO: update to better auth
    is_auth = username == USERNAME and password == PASSWORD
    logger.info(f"Log-in attempted. {is_auth=}")
    return is_auth


def chat(question, history):
    history = history or []
    answer = buster.process_input(question)

    # formatting hack for code blocks to render properly every time
    answer = answer.replace("```", "\n```\n")

    history.append((question, answer))
    return history, history


block = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")

with block:
    with gr.Row():
        gr.Markdown(
            "<h3><center>Buster 🤖: A Question-Answering Bot for your documentation</center></h3>"
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a question to AI stackoverflow here...",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    examples = gr.Examples(
        examples=[
            "How can I perform backpropagation?",
            "How do I deal with noisy data?",
        ],
        inputs=message,
    )

    gr.Markdown(
        "This application uses GPT to search the docs for relevant info and answer questions."
    )

    gr.HTML("️<center> Created with ❤️ by @jerpint and @hadrienbertrand")

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state], outputs=[chatbot, state])


block.launch(debug=True, share=False, auth=check_auth)
