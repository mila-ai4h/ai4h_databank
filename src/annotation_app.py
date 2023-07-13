from functools import lru_cache
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

top_k = 10
empty_documents = [""] * top_k
empty_evaluations = [False] * top_k


@lru_cache
def get_relevant_documents(
    query: str,
    top_k=10,
) -> pd.DataFrame:

    if query == "":
        return empty_documents

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
        question_input = gr.Textbox(
            label="Ask your question",
            placeholder="Ask your question here...",
            lines=1,
        )
        ask_button = gr.Button(value="Ask", variant="primary")
    with gr.Column(variant="panel"):
        gr.Markdown("## Example questions")
        with gr.Tab("Relevant questions"):
            gr.Examples(
                examples=relevant_questions,
                inputs=question_input,
                label="Questions users could ask.",
                # fn=ask_button.click,
                # run_on_click=True,
            )

    def update_documents(documents):
        updated_document_content = []

        for doc in documents:
            updated_document_content.append(doc)

        return updated_document_content

    # keep track of submission form components here...
    documents = gr.State(empty_documents)
    user_evaluations = gr.State({})

    document_evaluation = []
    document_content = []

    with gr.Row():
        with gr.Column():
            save_button = gr.Button(value="Save 💾", variant="primary")
            for idx in range(len(documents.value)):
                with gr.Column():
                    document_evaluation.append(gr.Checkbox(value=False, label="relevant", interactive=True))
                    document_content.append(
                        gr.Textbox(label=f"Document", interactive=False, value=documents.value[idx])
                    )

    def update_evaluations(question: str, user_evaluations):
        latest_evaluation = user_evaluations.get(question)
        if latest_evaluation is None:
            # return an empty evaluation
            print("Here")
            return empty_evaluations
        else:
            print("Here 1")

            return latest_evaluation

    def save_evaluations(question, user_evaluations, *current_evaluations):
        print(current_evaluations)
        user_evaluations[question] = list(current_evaluations)

        return user_evaluations

    # fmt: off
    ask_button.click(
        fn=get_relevant_documents, inputs=question_input, outputs=documents
    ).then(
        update_documents, inputs=[documents], outputs=document_content
    ).then(
        update_evaluations, inputs=[question_input, user_evaluations], outputs=document_evaluation
    )


    save_button.click(
        fn=save_evaluations, inputs=[question_input, user_evaluations, *document_evaluation], outputs=user_evaluations
    )
    # fmt: on
