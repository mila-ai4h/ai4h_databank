import cfg
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

retriever=cfg.retriever

query = "What is the best policy for AI"
matched_documents = retriever.retrieve(query=query, top_k=20)
