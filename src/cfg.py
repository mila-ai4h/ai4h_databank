import logging
import os

import openai
from buster.busterbot import Buster, BusterConfig
from buster.completers import ChatGPTCompleter, DocumentAnswerer
from buster.formatters.documents import DocumentsFormatter
from buster.formatters.prompts import PromptFormatter
from buster.retriever import Retriever, ServiceRetriever
from buster.tokenizers import Tokenizer
from buster.validators import QuestionAnswerValidator, Validator

from src.app_utils import init_db, make_uri

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WordTokenizer(Tokenizer):
    """Naive word-level tokenizer

    The original tokenizer from openAI eats way too much Ram.
    This is a naive word count tokenizer to be used instead."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, string):
        return string.split()

    def decode(self, encoded):
        return " ".join(encoded)


# set openAI creds
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

# set pinecone creds
pinecone_api_key = os.getenv("AI4H_PINECONE_API_KEY")
pinecone_env = os.getenv("AI4H_PINECONE_ENV")
pinecone_index = os.getenv("AI4H_PINECONE_INDEX")

# set mongo creds
mongo_username = os.getenv("AI4H_MONGODB_USERNAME")
mongo_password = os.getenv("AI4H_MONGODB_PASSWORD")
mongo_cluster = os.getenv("AI4H_MONGODB_CLUSTER")
mongo_uri = make_uri(mongo_username, mongo_password, mongo_cluster)
mongo_db_name = os.getenv("AI4H_MONGODB_DB_DATA")

username = os.getenv("AI4H_MONGODB_USERNAME")
password = os.getenv("AI4H_MONGODB_PASSWORD")
cluster = os.getenv("AI4H_MONGODB_CLUSTER")
db_name = os.getenv("AI4H_MONGODB_DB_NAME")
mongo_db = init_db(username, password, cluster, db_name)

mongo_feedback_collection = os.getenv("AI4H_MONGODB_FEEDBACK_COLLECTION")
mongo_arena_collection = os.getenv("AI4H_MONGODB_ARENA_COLLECTION")
mongo_interaction_collection = os.getenv("AI4H_MONGODB_INTERACTION_COLLECTION")


buster_cfg = BusterConfig(
    validator_cfg={
        "unknown_response_templates": [
            "I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
            "I cannot answer this question based on the information I have available",
            "The provided documents do not contain information on your given topic",
            "The provided documents do not directly address the question",
            "The provided documents do not directly address the question. I cannot answer this question based on the information I have available.",
        ],
        "unknown_threshold": 0.84,
        "embedding_model": "text-embedding-ada-002",
        "use_reranking": True,
        "invalid_question_response": "This question does not seem relevant to AI policies.",
        "check_question_prompt": """You are a chatbot answering questions on behalf of the OECD specifically on AI policies.
Your first job is to determine whether or not a question is valid, and should be answered.
For a question to be considered valid, it must be related to AI and policies.
More general questions are not considered valid, even if you might know the response.
A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.

For example:
Q: What policies did countries like Canada put in place with respect to artificial intelligence?
true

Q: What policies are put in place to ensure the wellbeing of agriculture?
false

Q:
""",
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "temperature": 0,
        },
    },
    retriever_cfg={
        "pinecone_api_key": pinecone_api_key,
        "pinecone_env": pinecone_env,
        "pinecone_index": pinecone_index,
        "mongo_uri": mongo_uri,
        "mongo_db_name": mongo_db_name,
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 3000,
        "embedding_model": "text-embedding-ada-002",
    },
    documents_answerer_cfg={
        "no_documents_message": "No documents are available for this question.",
    },
    completion_cfg={
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": True,
            "temperature": 0,
        },
    },
    tokenizer_cfg={
        "model_name": "gpt-3.5-turbo",
    },
    documents_formatter_cfg={
        "max_tokens": 3500,
        "formatter": "{content}",
    },
    prompt_formatter_cfg={
        "max_tokens": 3500,
        "text_before_docs": (
            "You are a chatbot assistant answering questions about artificial intelligence (AI) policies and laws. "
            "You represent the OECD AI Policy Observatory. "
            "You can only respond to a question if the content necessary to answer the question is contained in the following provided documents. "
            "If the answer is in the documents, summarize it in a helpful way to the user. "
            "If it isn't, simply reply that you cannot answer the question. "
            "Do not refer to the documents directly, but use the information provided within it to answer questions. "
            "Do not say 'according to the documentation' or related phrases. "
            "Here is the documentation:\n"
            "<DOCUMENTS> "
        ),
        "text_after_docs": (
            "<\\DOCUMENTS>\n"
            "REMEMBER:\n"
            "You are a chatbot assistant answering questions about artificial intelligence (AI) policies and laws. "
            "You represent the OECD AI Policy Observatory. "
            "Here are the rules you must follow:\n"
            "1) You must only respond with information contained in the documents above. Say you do not know if the information is not provided.\n"
            "2) Make sure to format your answers in Markdown format, including code block and snippets.\n"
            "3) Do not reference any links, urls or hyperlinks in your answers.\n"
            "4) Do not refer to the documentation directly, but use the information provided within it to answer questions.\n"
            "5) Do not say 'according to the documentation' or related phrases.\n"
            "6) If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:\n"
            "'I'm sorry, but I am an AI language model trained to assist with questions related to AI policies and laws. I cannot answer that question as it is not relevant to AI policies and laws. Is there anything else I can assist you with?'\n"
            "For example:\n"
            "Q: What is the meaning of life for a qa bot?\n"
            "A: I'm sorry, but I am an AI language model trained to assist with questions related to AI policies and laws. I cannot answer that question as it is not relevant to AI policies and laws. Is there anything else I can assist you with?\n"
            "7) If the provided documents do not directly address the question, simply state that the provided documents don't answer the question. Do not summarize what they do contain. "
            "For example: 'I cannot answer this question based on the information I have available'."
            "Now answer the following question:\n"
        ),
    },
)


def setup_buster(buster_cfg):
    retriever: Retriever = ServiceRetriever(**buster_cfg.retriever_cfg)
    tokenizer = WordTokenizer(**buster_cfg.tokenizer_cfg)
    # tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    document_answerer: DocumentAnswerer = DocumentAnswerer(
        completer=ChatGPTCompleter(**buster_cfg.completion_cfg),
        documents_formatter=DocumentsFormatter(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
        **buster_cfg.documents_answerer_cfg,
    )
    validator: Validator = QuestionAnswerValidator(**buster_cfg.validator_cfg)

    buster: Buster = Buster(retriever=retriever, document_answerer=document_answerer, validator=validator)
    return buster
