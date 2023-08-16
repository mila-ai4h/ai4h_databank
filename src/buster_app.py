import copy
import logging
import os

import gradio as gr
import pandas as pd
from buster.completers import Completion

import cfg
from cfg import setup_buster
from feedback import FeedbackForm, Interaction
from src.app_utils import add_sources, check_auth, get_utc_time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mongo_db = cfg.mongo_db
buster_cfg = copy.deepcopy(cfg.buster_cfg)
buster = setup_buster(buster_cfg=buster_cfg)
max_sources = cfg.buster_cfg.retriever_cfg["top_k"]
from pathlib import Path

current_dir = Path(__file__).resolve().parent

# get documents metadata
documents_metadata_file = str(current_dir / "documents_metadata.csv")
documents_metadata = pd.read_csv(documents_metadata_file)

# Load the sample questions and split them by type
questions_file = str(current_dir / "sample_questions.csv")
questions = pd.read_csv(questions_file)
relevant_questions = questions[questions.question_type == "relevant"].question.to_list()
irrelevant_questions = questions[questions.question_type == "irrelevant"].question.to_list()
trick_questions = questions[questions.question_type == "trick"].question.to_list()


def get_metadata_markdown(df):
    metadata = []

    def to_link(title: str, link: str):
        return f"[{title}]({link})"

    for _, item in df.iterrows():
        # metadata.append(" | ".join([str(i) for i in item]))

        source = item["Source"]
        link = item["Link"]
        title = item["Title"]  # limit to 50 chars
        year = item["Year"]

        metadata.append(f"{source} | {year} | {to_link(title, link)} ")
    metadata_str = "\n".join(metadata)

    markdown_text = f"""
## 📚 Sources
| Source | Year | Report |
| ---    | ---    | --- |
{metadata_str}
"""
    return markdown_text


def append_completion(completion, user_completions):
    user_completions.append(completion)
    return user_completions


def user(user_input, history):
    """Adds user's question immediately to the chat."""
    return "", history + [[user_input, None]]


def chat(history):
    user_input = history[-1][0]

    completion = buster.process_input(user_input)

    history[-1][1] = ""

    for token in completion.answer_generator:
        history[-1][1] += token

        yield history, completion


def log_completion(
    completion: Completion,
    request: gr.Request,
):
    collection = cfg.mongo_interaction_collection

    interaction = Interaction(
        user_completions=[completion],
        time=get_utc_time(),
        username=request.username,
    )
    interaction.send(mongo_db, collection=collection)


def submit_feedback(
    user_completions,
    feedback_relevant_sources,
    feedback_relevant_answer,
    feedback_info,
    request: gr.Request,
):
    feedback_form = FeedbackForm(
        extra_info=feedback_info,
        relevant_answer=feedback_relevant_answer,
        relevant_sources=feedback_relevant_sources,
    )
    feedback = Interaction(
        user_completions=user_completions,
        form=feedback_form,
        time=get_utc_time(),
        username=request.username,
    )
    feedback.send(mongo_db, collection=cfg.mongo_feedback_collection)


def toggle_feedback_visible(visible: bool):
    """Toggles the visibility of the 'feedback submitted' message."""
    return {feedback_submitted_message: gr.update(visible=visible)}


def clear_feedback_form():
    """Clears the contents of the feedback form."""
    return {
        feedback_submitted_message: gr.update(visible=False),
        feedback_relevant_sources: gr.update(value=None),
        feedback_relevant_answer: gr.update(value=None),
        feedback_info: gr.update(value=""),
    }


buster_app = gr.Blocks()

with buster_app:
    # TODO: trigger a proper change to update

    # state variables are client-side and are reset every time a client refreshes the page
    user_completions = gr.State([])

    app_name = "LLaWma 🦙"
    gr.Markdown(f"<h1><center>{app_name}: A Question-Answering Bot for your documentation</center></h1>")
    with gr.Row(variant="panel"):
        with gr.Box():
            gr.Markdown(
                f"""

            ## Welcome!

            {app_name} is connected to AI policies from various sources. Using this platform, you can ask AI-policy questions and {app_name} will look for the most relevant policies to formulate an answer based on the sources.

            ## How it works
            This app uses language models to convert documents to their semanatic representations.
            When a user asks a question, {app_name} compares it against all available documents. It then retrieves the documents that are most relevance to the question and prompts ChatGPT with those documents to generate a response.
            The answer and accompanying sources are then displayed to the user.
            """
            )

        with gr.Box():
            gr.Markdown(
                f"""
            ## Limitations

            {app_name} is intended to ***_only be used as a demo._*** While we have worked hard to make this as useful as possible, it is important to understand that there are no guarantees regarding the accuracy of its responses.
            Like all language models, {app_name} might generate information that is not entirely reliable. To mitigate this, users are strongly advised to independently verify the information provided by the tool.
            All sources available to the model are listed below.

            ## Recommended usage

            For optimal results, employ {app_name} in scenarios where the answers to questions are explicitly present within the provided documentation.
            It is most effective for queries that require direct extraction of information. However, for questions that demand complex reasoning spanning across an entire document or require contextual understanding, the performance of {app_name} might be limited. In such cases, alternative methods of information retrieval and analysis might be more appropriate.

            """
            )

    with gr.Row():
        with gr.Column(scale=2, variant="panel"):
            gr.Markdown("## Chatbot")
            chatbot = gr.Chatbot()
            message = gr.Textbox(
                label="Chat with 🦙",
                placeholder="Ask your question here...",
                lines=1,
            )
            submit = gr.Button(value="Send", variant="primary")

            with gr.Column(variant="panel"):
                gr.Markdown("## Example questions")
                with gr.Tab("Relevant questions"):
                    gr.Examples(
                        examples=relevant_questions[0:5],
                        inputs=message,
                        label="Questions users could ask.",
                    )
                with gr.Tab("Irrelevant questions"):
                    gr.Examples(
                        examples=irrelevant_questions,
                        inputs=message,
                        label="Questions with no relevance to the OECD AI Policy Observatory.",
                    )
                with gr.Tab("Trick questions"):
                    gr.Examples(
                        examples=trick_questions,
                        inputs=message,
                        label="Questions about non-existing AI policies and laws.",
                    )

        with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown(
                    """## Relevant Documents
                All retrieved documents will be listed here in order of importance. If no answer was found, documents will not be displayed.
                """
                )
                sources_textboxes = []
                for i in range(max_sources):
                    with gr.Tab(f"Source {i + 1} 📝"):
                        t = gr.Markdown()
                    sources_textboxes.append(t)

            with gr.Column():
                gr.Markdown("## Parameters")
                metadata = [
                    ("generation model", cfg.buster_cfg.completion_cfg["completion_kwargs"]["model"]),
                    ("embedding model", cfg.buster_cfg.retriever_cfg["embedding_model"]),
                ]
                gr.HighlightedText(value=metadata, label="Parameters")

    # Feedback
    with gr.Box():
        with gr.Row():
            with gr.Box():
                with gr.Column():
                    gr.Markdown("## Feedback Form")
                    with gr.Row():
                        feedback_relevant_sources = gr.Radio(
                            choices=["👍", "👎"], label="Were any of the retrieved sources relevant?"
                        )
                    with gr.Row():
                        feedback_relevant_answer = gr.Radio(
                            choices=["👍", "👎"], label="Was the generated answer useful?"
                        )

                    feedback_info = gr.Textbox(
                        label="Enter additional information (optional)",
                        lines=10,
                        placeholder="Enter more helpful information for us here...",
                    )

                    submit_feedback_btn = gr.Button("Submit Feedback!")
                    with gr.Column(visible=False) as feedback_submitted_message:
                        gr.Markdown("Feedback recorded, thank you! 📝")
            with gr.Box():
                with gr.Column():
                    gr.Markdown(
                        f"""## Help Us Improve with Your Feedback! 👍👎
By filling out the feedback form, you're helping us understand what's working well and where we can make improvements on {app_name}.

Rest assured, your feedback is completely anonymous, and we don't collect any personal information. This means you can express your thoughts openly and honestly, allowing us to gain valuable insights into how we can enhance the chatbot's performance and accuracy.

Every thumbs up or thumbs down helps us determine how to make the chatbot better. Thank you for contributing to the evolution of our app.

"""
                    )

    # TODO: Pick how to display the sources, 2 options for now
    # Display the sources using a dataframe...
    # gr.DataFrame(documents_metadata, headers=list(documents_metadata.columns), interactive=False)

    with gr.Box():
        with gr.Column():
            # ... Or display the sources using markdown.
            gr.Markdown(get_metadata_markdown(documents_metadata))

            gr.HTML("<center> Powered by <a href='https://github.com/jerpint/buster'>Buster</a> 🤖</center>")

    # store the users' last completion here
    completion = gr.State()

    # fmt: off
    submit_feedback_btn.click(
        toggle_feedback_visible,
        inputs=gr.State(False),
        outputs=feedback_submitted_message,
    ).then(
        submit_feedback,
        inputs=[
            user_completions,
            feedback_relevant_sources,
            feedback_relevant_answer,
            feedback_info,
        ],
    ).success(
        toggle_feedback_visible,
        inputs=gr.State(True),
        outputs=feedback_submitted_message,
    )
    # If you rage click the subimt feedback button, it re-appears so you are confident it was recorded properly.
    # fmt: on


    # fmt: off
    submit.click(
        user, [message, chatbot], [message, chatbot]
    ).then(
        clear_feedback_form,
        outputs=[feedback_submitted_message, feedback_relevant_sources, feedback_relevant_answer, feedback_info]
    ).then(
        chat,
        inputs=[chatbot],
        outputs=[chatbot, completion],
    ).then(
        add_sources,
        inputs=[completion, gr.State(max_sources)],
        outputs=[*sources_textboxes]
    ).then(
        log_completion,
        inputs=completion,
    ).then(
        append_completion,
        inputs=[completion, user_completions], outputs=[user_completions]
    )
    message.submit(
        user, [message, chatbot], [message, chatbot]
    ).then(
        clear_feedback_form,
        outputs=[feedback_submitted_message, feedback_relevant_sources, feedback_relevant_answer, feedback_info]
    ).then(
        chat,
        inputs=[chatbot],
        outputs=[chatbot, completion],
    ).then(
        add_sources,
        inputs=[completion, gr.State(max_sources)],
        outputs=[*sources_textboxes]
    ).then(
        log_completion,
        inputs=completion,
    ).then(
        append_completion,
        inputs=[completion, user_completions], outputs=[user_completions]
    )
    # fmt: on


# True when launching using gradio entrypoint
if os.getenv("MOUNT_GRADIO_APP") is None:
    logger.info("launching app via gradio")
    buster_app.queue(concurrency_count=16)
    buster_app.launch(share=False, auth=check_auth)
