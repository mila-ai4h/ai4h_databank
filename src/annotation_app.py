import cfg
import logging
import gradio as gr
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

retriever = cfg.retriever

# query = "What is the best policy for AI"
# matched_documents = retriever.retrieve(query=query, top_k=20)

# Load the sample questions and split them by type
questions = pd.read_csv("sample_questions.csv")
relevant_questions = questions[questions.question_type == "relevant"].question.to_list()

annotation_app = gr.Blocks()


def get_relevant_documents(
    query: str,
    top_k=10,
) -> pd.DataFrame:
    # query = "What is the best policy for AI"
    matched_documents = retriever.retrieve(query=query, top_k=top_k)

    return matched_documents.content.to_list()


def get_document(idx, documents):
    return documents[idx]


with annotation_app:
    # TODO: trigger a proper change to update

    # state variables are client-side and are reset every time a client refreshes the page
    user_responses = gr.State([])

    gr.Markdown("<h1><center>Reference Annotation</center></h1>")

    with gr.Column(scale=2):
        message = gr.Textbox(
            label="Ask your question",
            placeholder="Ask your question here...",
            lines=1,
        )
        ask_button = gr.Button(value="Ask", variant="secondary").style(full_width=False)
    with gr.Column(variant="panel"):
        gr.Markdown("## Example questions")
        with gr.Tab("Relevant questions"):
            gr.Examples(
                examples=relevant_questions,
                inputs=message,
                label="Questions users could ask.",
            )

    def update_documents(*args):
        documents = args[0]
        document_content = args[1:]
        updated_document_content = []

        for doc in documents:
            updated_document_content.append(doc)

        return updated_document_content

    # keep track of submission form components here...
    top_k = 10
    documents = gr.State([""] * top_k)
    document_relevance = []
    document_content = []

    with gr.Row():
        with gr.Column():
            for idx in range(len(documents.value)):
                with gr.Column():
                    document_relevance.append(gr.Checkbox(value=False, label="relevant", interactive=True))
                    document_content.append(
                        gr.Textbox(label=f"Document", interactive=False, value=documents.value[idx])
                    )

    ask_button.click(fn=get_relevant_documents, inputs=message, outputs=documents).then(
        update_documents, inputs=[documents, *document_content], outputs=document_content
    )
