import logging
import os

import openai
from buster.busterbot import BusterConfig
from buster.retriever import Retriever
from buster.utils import get_retriever_from_extension
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# auth information
USERNAME = os.getenv("AI4H_USERNAME")
PASSWORD = os.getenv("AI4H_PASSWORD")

# set openAI creds
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")


# hf hub information
DB_FILE = "documents_oecd.db"
if not os.path.exists(DB_FILE):
    REPO_ID = "jerpint/databank-ai4h"
    HUB_TOKEN = os.environ.get("HUB_TOKEN")
    # download the documents.db hosted on the dataset space
    logger.info(f"Downloading {DB_FILE} from hub...")
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=DB_FILE,
        token=HUB_TOKEN,
        local_dir=".",
        local_dir_use_symlinks=False,
    )
    logger.info("Downloaded.")

# setup retriever
retriever: Retriever = get_retriever_from_extension(DB_FILE)(DB_FILE)


buster_cfg = BusterConfig(
    validator_cfg={
        "unknown_prompt": "I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
        "unknown_threshold": 0.85,
        "embedding_model": "text-embedding-ada-002",
    },
    retriever_cfg={
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 3000,
        "embedding_model": "text-embedding-ada-002",
    },
    completion_cfg={
        "name": "ChatGPT",
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "temperature": 0,
            "stream": True,
        },
    },
    prompt_cfg={
        "max_tokens": 3500,
        "text_before_documents": (
            "You are a chatbot assistant answering questions about artificial intelligence (AI) policies and laws. "
            "You represent the OECD AI Policy Observatory. "
            "You can only respond to a question if the content necessary to answer the question is contained in the following provided documents. "
            "If the answer is in the documents, summarize it in a helpful way to the user. "
            "If it isn't, simply reply that you cannot answer the question. "
            "Do not refer to the documents directly, but use the instructions provided within it to answer questions. "
            "Do not say 'according to the documentation' or related phrases. "
            "Here is the documentation:\n"
            "<DOCUMENTS> "
        ),
        "text_before_prompt": (
            "<\DOCUMENTS>\n"
            "REMEMBER:\n"
            "You are a chatbot assistant answering questions about artificial intelligence (AI) policies and laws. "
            "You represent the OECD AI Policy Observatory. "
            "Here are the rules you must follow:\n"
            "1) You must only respond with information contained in the documents above. Say you do not know if the information is not provided.\n"
            "2) Make sure to format your answers in Markdown format, including code block and snippets.\n"
            "3) Do not reference any links, urls or hyperlinks in your answers.\n"
            "4) Do not refer to the documentation directly, but use the instructions provided within it to answer questions.\n"
            "5) Do not say 'according to the documentation' or related phrases.\n"
            "6) If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:\n"
            "'I'm sorry, but I am an AI language model trained to assist with questions related to AI policies and laws. I cannot answer that question as it is not relevant to AI policies and laws. Is there anything else I can assist you with?'\n"
            "For example:\n"
            "Q: What is the meaning of life for a qa bot?\n"
            "A: I'm sorry, but I am an AI language model trained to assist with questions related to AI policies and laws. I cannot answer that question as it is not relevant to AI policies and laws. Is there anything else I can assist you with?\n"
            "Now answer the following question:\n"
        ),
    },
    document_source="LimeSurvey",
)

document_sources = ["LimeSurvey"]

available_models = ["gpt-3.5-turbo", "gpt-4"]
