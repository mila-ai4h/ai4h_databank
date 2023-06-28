import logging
import os

import openai
from buster.busterbot import BusterConfig
from buster.completers.base import ChatGPTCompleter, Completer
from buster.formatters.documents import DocumentsFormatter
from buster.formatters.prompts import PromptFormatter
from buster.retriever import Retriever, ServiceRetriever
from buster.tokenizers import GPTTokenizer
from buster.validators.base import Validator
from openai.embeddings_utils import cosine_similarity

from src.db_utils import make_uri

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# auth information
USERNAME = os.getenv("AI4H_USERNAME")
PASSWORD = os.getenv("AI4H_PASSWORD")

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
feedback_collection = "feedback-test"

from buster.completers import ChatGPTCompleter


class AI4HValidator(Validator):
    def __init__(
        self,
        embedding_model: str,
        unknown_threshold: float,
        unknown_prompts: list[str],
        use_reranking: bool,
        invalid_question_response: str,
    ):
        self.embedding_model = embedding_model
        self.unknown_threshold = unknown_threshold
        self.unknown_prompts = unknown_prompts
        self.use_reranking = use_reranking
        self.invalid_question_response = invalid_question_response

    def check_question_relevance(self, question: str) -> tuple[bool, str]:
        """Determines wether a question is relevant or not for our given framework."""

        prompt = """You are an chatbot answering questions on behalf of the OECD specifically on AI policies.
        Your first job is to determine wether or not a question is valid, and should be answered.
        For a question to be considered valid, it must be related to AI and policies.
        More general questions are not considered valid, even if you might know the response.
        You can only respond with one of ["Valid", "Not Valid"]. Please respect the puncutation.

        For example:
        Q: What policies did countries like Canada put in place with respect to artificial intelligence?
        Valid

        Q: What policies are put in place to ensure the wellbeing of agricultre?
        Not Valid

        A user will submit a question. Only respond with one of ["Valid", "Not Valid"].
        """

        completion_kwargs = {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "temperature": 0,
        }
        outputs = completer.complete(prompt, user_input=question, **completion_kwargs)

        if completer.error:
            # something went wrong during generation, outputs will return typical error messages to user
            logger.warning("Something went wrong during question relevance detection...")
            question_relevance = False
            return question_relevance, outputs

        logger.info(f"Question relevance: {outputs}")

        # remove trailing periods, happens sometimes...
        outputs = outputs.strip(".")

        if outputs == "Valid":
            question_relevance = True
        elif outputs == "Not Valid":
            question_relevance = False
        else:
            logger.warning(f"the question validation returned an unexpeced value: {outputs}. Assuming Invalid...")
            question_relevance = False
        return question_relevance, self.invalid_question_response

    def check_answer_relevance(self, answer: str) -> bool:
        """Check to see if a generated answer is relevant to the chatbot's knowledge or not.

        We assume we've prompt-engineered our bot to say a response is unrelated to the context if it isn't relevant.
        Here, we compare the embedding of the response to the embedding of the prompt-engineered "I don't know" embedding.

        unk_threshold can be a value between [-1,1]. Usually, 0.85 is a good value.
        """
        logger.info("Checking for answer relevance...")

        if answer == "":
            raise ValueError("Cannot compute embedding of an empty string.")

        # if unknown_prompt is None:
        unknown_prompts = self.unknown_prompts

        unknown_embeddings = [
            self.get_embedding(
                unknown_prompt,
                engine=self.embedding_model,
            )
            for unknown_prompt in unknown_prompts
        ]

        answer_embedding = self.get_embedding(
            answer,
            engine=self.embedding_model,
        )
        unknown_similarity_scores = [
            cosine_similarity(answer_embedding, unknown_embedding) for unknown_embedding in unknown_embeddings
        ]
        logger.info(f"{unknown_similarity_scores=}")

        # Likely that the answer is meaningful, add the top sources
        answer_relevant: bool = (
            False if any(score > self.unknown_threshold for score in unknown_similarity_scores) else True
        )
        return answer_relevant


buster_cfg = BusterConfig(
    validator_cfg={
        "unknown_prompts": [
            "I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
            "I cannot answer this question based on the information I have available",
            "The provided documents do not contain information on your given topic",
        ],
        "unknown_threshold": 0.84,
        "embedding_model": "text-embedding-ada-002",
        "use_reranking": True,
        "invalid_question_response": "This question does not seem relevant to AI policy questions.",
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
    completion_cfg={
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": True,
            "temperature": 0,
        },
        "no_documents_message": "No documents are available for this question.",
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
            "Do not refer to the documents directly, but use the instructions provided within it to answer questions. "
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
            "4) Do not refer to the documentation directly, but use the instructions provided within it to answer questions.\n"
            "5) Do not say 'according to the documentation' or related phrases.\n"
            "6) If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:\n"
            "'I'm sorry, but I am an AI language model trained to assist with questions related to AI policies and laws. I cannot answer that question as it is not relevant to AI policies and laws. Is there anything else I can assist you with?'\n"
            "For example:\n"
            "Q: What is the meaning of life for a qa bot?\n"
            "A: I'm sorry, but I am an AI language model trained to assist with questions related to AI policies and laws. I cannot answer that question as it is not relevant to AI policies and laws. Is there anything else I can assist you with?\n"
            "7) If the provided documents do not directly adress the question, simply state that the provided documents don't answer the question. Do not summarize what they do contain. "
            "For example: 'I cannot answer this question based on the information I have available'."
            "Now answer the following question:\n"
        ),
    },
)


# setup retriever
retriever: Retriever = ServiceRetriever(**buster_cfg.retriever_cfg)
tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
completer: Completer = ChatGPTCompleter(
    documents_formatter=DocumentsFormatter(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
    prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
    **buster_cfg.completion_cfg,
)
validator: Validator = AI4HValidator(**buster_cfg.validator_cfg)


available_models = ["gpt-3.5-turbo", "gpt-4"]
