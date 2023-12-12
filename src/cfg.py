import logging
import os
from pathlib import Path

import openai
from huggingface_hub import hf_hub_download

from buster.busterbot import Buster, BusterConfig
from buster.completers import ChatGPTCompleter, DocumentAnswerer
from buster.formatters.documents import DocumentsFormatterJSON
from buster.formatters.prompts import PromptFormatter
from buster.llm_utils import QuestionReformulator
from buster.llm_utils.embeddings import get_openai_embedding_constructor
from buster.retriever import DeepLakeRetriever, Retriever, ServiceRetriever
from buster.tokenizers import GPTTokenizer
from buster.utils import extract_zip
from buster.validators import Validator
from src.app_utils import get_logging_db_name, init_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Note: The app will not launch if the environment variables aren't set. This is intentional.
# Set OpenAI Configurations
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORG_ID"]

# Set relative path to data dir
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"  # ../data


# the embedding function that will get used throughout the app
embedding_fn = get_openai_embedding_constructor(
    model="text-embedding-ada-002", client_kwargs={"timeout": 2, "max_retries": 2}
)

CHUNKS_VERSION = "data-2023-11-02"

# Deeplake configuration
deeplake_dir = current_dir.parent / "deeplake_data"  # ../data
DEEPLAKE_VECTOR_STORE_PATH = os.path.join(deeplake_dir, CHUNKS_VERSION)
HF_TOKEN = os.environ["HF_TOKEN"]
HF_DATASET_REPO_ID = "mila-quebec/sai-data"
HF_VECTOR_STORE_PATH = CHUNKS_VERSION + ".zip"


def download_deeplake_vector_store():
    """Downloads the vector store stored on the huggingface dataset."""

    hf_hub_download(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename=HF_VECTOR_STORE_PATH,
        local_dir=".",
        local_dir_use_symlinks=False,
    )

    extract_zip(HF_VECTOR_STORE_PATH, DEEPLAKE_VECTOR_STORE_PATH)


# Pinecone Configurations
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = "asia-southeast1-gcp"
PINECONE_INDEX = "oecd"
PINECONE_NAMESPACE = CHUNKS_VERSION

# MongoDB Configurations
MONGO_URI = os.environ["MONGO_URI"]

# Instance Configurations
INSTANCE_NAME = os.environ["INSTANCE_NAME"]  # e.g., huggingface, heroku
INSTANCE_TYPE = os.environ["INSTANCE_TYPE"]  # e.g. ["dev", "prod", "local"]

# MongoDB Databases
MONGO_DATABASE_LOGGING = get_logging_db_name(INSTANCE_TYPE)  # Where all interactions will be stored
MONGO_DATABASE_DATA = CHUNKS_VERSION  # Where documents are stored

# Check that data chunks are aligned on Mongo and Pinecone
if MONGO_DATABASE_DATA != PINECONE_NAMESPACE:
    logger.warning(
        f"""The collection is different on pinecone and Mongo, is this expected?

        {MONGO_DATABASE_DATA=}
        {PINECONE_NAMESPACE=}
        """
    )

# MongoDB Collections
# Naming convention: Collection name followed by purpose.
MONGO_COLLECTION_FEEDBACK = "feedback"  # Feedback form
MONGO_COLLECTION_INTERACTION = "interaction"  # User interaction
MONGO_COLLECTION_FLAGGED = "flagged"  # Flagged interactions

# Make the connections to the databases
mongo_db = init_db(mongo_uri=MONGO_URI, db_name=MONGO_DATABASE_LOGGING)


app_name = "SAI ️💬"

# User settings default values
reveal_user_settings = False  # Wheter to display settings to the user or not
max_sources = 15  # maximum number of sources that can be set by a user for retrieval
reformulate_question = False  # Default setting for reformulating a user's question

# sample questions
example_questions = [
    "Are there any AI policies related to AI adoption in the public sector in the UK?",
    "How is Canada evaluating the success of its AI strategy?",
    "Has the EU proposed specific legislation on AI?",
]


disclaimer = f"""
**Use the feedback form on the right to help us improve** 👉

**Always verify the integrity of {app_name} responses using the sources provided below** 👇
"""

message_before_reformulation = "I reformulated your answer to: '"
message_after_reformulation = (
    "'\n\nThis is done automatically to increase performance of the tool. You can disable this in the Settings ⚙️ tab."
)

# default client config for OpenAI Completions
client_kwargs = {
    "api_key": os.environ["OPENAI_API_KEY"],
    "organization": os.environ["OPENAI_ORG_ID"],
    "timeout": 10,
    "max_retries": 2,
}


buster_cfg = BusterConfig(
    validator_cfg={
        "use_reranking": True,  # Reranks documents according to generated answer
        "validate_documents": False,  # Validates documents using chatGPT (expensive)
        "answer_validator_cfg": {
            "unknown_response_templates": [
                "I cannot answer this question based on the information I have available",
                "The information I have access to does not address the question",
            ],
            "unknown_threshold": 0.84,
            "embedding_fn": embedding_fn,
        },
        "question_validator_cfg": {
            "invalid_question_response": """Thank you for your question! Unfortunately, I haven't been able to find the information you're looking for. Your question might be:
                    * Outside the scope of AI policy documents
                    * Too recent (i.e. draft policies) or about the future
                    * Building on my previous answer (I have no memory of previous conversations)
                    * Vague (i.e not affiliated with a specific country)
                    * Asking the model to perform its own assessment of the policies (i.e. What is the best/worst AI policy)
                    You can always try rewording your question and ask again!
                    """,
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
                "model": "ft:gpt-3.5-turbo-0613:oecd-ai:first-finetune:8LEyi8pG",
                "stream": False,
                "temperature": 0,
            },
            "client_kwargs": client_kwargs,
        },
    },
    retriever_cfg={
        # deeplake cfg
        "path": DEEPLAKE_VECTOR_STORE_PATH,
        "top_k": 3,
        "thresh": 0.7,
        "embedding_fn": embedding_fn,
    },
    # retriever_cfg={
    #     # service retriever cfg
    #     "pinecone_api_key": PINECONE_API_KEY,
    #     "pinecone_env": PINECONE_ENV,
    #     "pinecone_index": PINECONE_INDEX,
    #     "pinecone_namespace": PINECONE_NAMESPACE,
    #     "mongo_uri": MONGO_URI,
    #     "mongo_db_name": MONGO_DATABASE_DATA,
    #     "top_k": 3,
    #     "thresh": 0.7,
    #     "embedding_fn": embedding_fn,
    # },
    documents_answerer_cfg={
        "no_documents_message": "No documents are available for this question.",
    },
    completion_cfg={
        "completion_kwargs": {
            "model": "gpt-3.5-turbo-0613",
            "stream": True,
            "temperature": 0,
        },
        "client_kwargs": client_kwargs,
    },
    tokenizer_cfg={
        "model_name": "gpt-3.5-turbo-0613",
    },
    documents_formatter_cfg={
        "max_tokens": 3500,
        "columns": ["content", "source", "title"],
    },
    question_reformulator_cfg={
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "temperature": 0,
        },
        "system_prompt": """
        Your role is to reformat a user's input into a question that is useful in the context of a semantic retrieval system.
        Reformulate the question in a way that captures the original essence of the question while also adding more relevant details that can be useful in the context of semantic retrieval.""",
    },
    prompt_formatter_cfg={
        "max_tokens": 4000,
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
            "9) Keep a neutral tone, and put things into context.\n"
            "For example:\n"
            "Q: What do African countries say about data privacy?\n"
            "A: There are currently 28 countries in Africa that have personal data protection legislation. However, limited resources, a lack of clear leadership in the region and localized approaches with regard to data-driven technology run the risk of creating an unfavorable environment for data privacy regulation from a business and data rights standpoint. For example, without policies for data sharing across countries, multinational data companies may choose to move their foreign direct investment to more favorable destinations. African countries also grapple with issues of higher priority, which stunts progress in the field of data privacy. According to one regional policy expert, “a government that is still battling [to set up a] school feeding programme in 2019 is not going to be the one to prioritise data and data protection policies with respect to AI.”\n"
            "Now answer the following question:\n"
        ),
    },
)


def setup_buster(buster_cfg):
    download_deeplake_vector_store()
    # retriever: Retriever = ServiceRetriever(**buster_cfg.retriever_cfg)
    retriever: Retriever = DeepLakeRetriever(**buster_cfg.retriever_cfg)
    tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    document_answerer: DocumentAnswerer = DocumentAnswerer(
        completer=ChatGPTCompleter(**buster_cfg.completion_cfg),
        documents_formatter=DocumentsFormatterJSON(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
        **buster_cfg.documents_answerer_cfg,
    )
    validator = Validator(**buster_cfg.validator_cfg)

    question_reformulator = QuestionReformulator(**buster_cfg.question_reformulator_cfg)

    buster: Buster = Buster(
        retriever=retriever,
        document_answerer=document_answerer,
        validator=validator,
        question_reformulator=question_reformulator,
    )
    return buster
