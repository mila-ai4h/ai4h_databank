import logging
import os
from pathlib import Path

import openai

from buster.busterbot import Buster, BusterConfig
from buster.completers import ChatGPTCompleter, DocumentAnswerer
from buster.formatters.documents import DocumentsFormatterJSON
from buster.formatters.prompts import PromptFormatter
from buster.retriever import Retriever, ServiceRetriever
from buster.validators import QuestionAnswerValidator, Validator
from src.app_utils import WordTokenizer, get_logging_db_name, init_db, make_uri

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Note: The app will not launch if the environment variables aren't set. This is intentional.
# Set OpenAI Configurations
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORGANIZATION"]

# Pinecone Configurations
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = "asia-southeast1-gcp"
PINECONE_INDEX = "oecd"
PINECONE_NAMESPACE = "data-2023-05-16"

# MongoDB Configurations
MONGO_URI = os.environ["MONGO_URI"]

# Instance Configurations
INSTANCE_NAME = os.environ["INSTANCE_NAME"]  # e.g., huggingface, heroku
INSTANCE_TYPE = os.environ["INSTANCE_TYPE"]  # e.g. ["dev", "prod", "local"]

# MongoDB Databases
MONGO_DATABASE_LOGGING = get_logging_db_name(INSTANCE_TYPE)  # Where all interactions will be stored
MONGO_DATABASE_DATA = "data-2023-05-16"  # Where documents are stored

# MongoDB Collections
# Naming convention: Collection name followed by purpose.
MONGO_COLLECTION_FEEDBACK = "feedback"  # Feedback form
MONGO_COLLECTION_INTERACTION = "interaction"  # User interaction
MONGO_COLLECTION_FLAGGED = "flagged"  # Flagged interactions

# Make the connections to the databases
mongo_db = init_db(mongo_uri=MONGO_URI, db_name=MONGO_DATABASE_LOGGING)


# Set relative path to data dir
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"  # ../data

app_name = "AIPS 🐒"

# sample questions
example_questions = [
    "Are there any AI policies related to AI adoption in the public sector in the UK?",
    "How is Canada evaluating the success of its AI strategy?",
    "Has the EU proposed specific legislation on AI?",
]


disclaimer = """I'm a bot 🤖 and can sometimes give inaccurate information.
Always verify the integrity of my statements using the sources provided below 📝

If something doesn't look right, you can use the feedback form to help me improve 😇
"""

buster_cfg = BusterConfig(
    validator_cfg={
        "unknown_response_templates": [
            "I cannot answer this question based on the information I have available",
            "The information I have access to does not address the question",
        ],
        "unknown_threshold": 0.84,
        "embedding_model": "text-embedding-ada-002",
        "use_reranking": True,
        "invalid_question_response": "I cannot answer this question as it does not seem relevant to AI policies. If you believe this is a mistake, please provide feedback through the panel on the right side. You can also try reformulating your question for better results.",
        "check_question_prompt": """You are a chatbot answering questions on behalf of the OECD specifically on AI policies.
Your first job is to determine whether or not a question is valid, and should be answered.
For a question to be considered valid, it must be related to AI and policies.
More general questions are not considered valid, even if you might know the response.
A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.
Do not judge the tone of the question. As long as it is relevant to the topic, respond 'true'.

For example:
Q: What policies did countries like Canada put in place with respect to artificial intelligence?
true

Q: What policies are put in place to ensure the wellbeing of agriculture?
false

Q:
""",
        "completion_kwargs": {
            "model": "gpt-3.5-turbo-0613",
            "stream": False,
            "temperature": 0,
        },
    },
    retriever_cfg={
        "pinecone_api_key": PINECONE_API_KEY,
        "pinecone_env": PINECONE_ENV,
        "pinecone_index": PINECONE_INDEX,
        "pinecone_namespace": PINECONE_NAMESPACE,
        "mongo_uri": MONGO_URI,
        "mongo_db_name": MONGO_DATABASE_DATA,
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
            "model": "gpt-3.5-turbo-0613",
            "stream": True,
            "temperature": 0,
        },
    },
    tokenizer_cfg={
        "model_name": "gpt-3.5-turbo-0613",
    },
    documents_formatter_cfg={
        "max_tokens": 3500,
        "columns": ["content", "source", "title"],
    },
    prompt_formatter_cfg={
        "max_tokens": 3500,
        "text_before_docs": (
            "You are a chatbot assistant answering questions about artificial intelligence (AI) policies and laws. "
            "You represent the OECD AI Policy Observatory. "
            "You can only respond to a question if the content necessary to answer the question is contained in the information provided to you. "
            "The information will be provided in a json format. "
            "If the answer is found in the information provided, summarize it in a helpful way to the user. "
            "If it isn't, simply reply that you cannot answer the question. "
            "Do not refer to the documents directly, but use the information provided within it to answer questions. "
            "Always cite which document you pulled information from. "
            "Do not say 'according to the documentation' or related phrases. "
            "Do not mention the documents directly, but use the information available within them to answer the question. "
            "You are forbidden from using the expressions 'according to the documentation' and 'the provided documents'. "
            "Here is the information available to you in a json table:\n"
        ),
        "text_after_docs": (
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
            "7) Always cite which document you pulled information from. Do this directly in the text. You can refer directly to the title in-line with your answer. Make it clear when information came directly from a source. "
            "8) If the information available to you does not directly address the question, simply state that you do not have the information required to answer. Do not summarize what is available to you. "
            "For example, say: 'I cannot answer this question based on the information I have available.'\n"
            "Now answer the following question:\n"
        ),
    },
)


def setup_buster(buster_cfg):
    retriever: Retriever = ServiceRetriever(**buster_cfg.retriever_cfg)
    tokenizer = WordTokenizer(**buster_cfg.tokenizer_cfg)
    document_answerer: DocumentAnswerer = DocumentAnswerer(
        completer=ChatGPTCompleter(**buster_cfg.completion_cfg),
        documents_formatter=DocumentsFormatterJSON(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
        **buster_cfg.documents_answerer_cfg,
    )
    validator: Validator = QuestionAnswerValidator(**buster_cfg.validator_cfg)

    buster: Buster = Buster(retriever=retriever, document_answerer=document_answerer, validator=validator)
    return buster
