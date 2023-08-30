import copy
import logging
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
from buster.completers import Completion

import cfg
from cfg import setup_buster
from feedback import FeedbackForm, Interaction
from src.app_utils import add_sources, check_auth, get_utc_time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Typehint for chatbot history
ChatHistory = list[list[Optional[str], Optional[str]]]

mongo_db = cfg.mongo_db
buster_cfg = copy.deepcopy(cfg.buster_cfg)
buster = setup_buster(buster_cfg=buster_cfg)
max_sources = cfg.buster_cfg.retriever_cfg["top_k"]

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

enable_terms_and_conditions = True

app_name = "LLaWma 🦙"


def toggle_additional_feedback():
    return {feedback_elems["show_additional_feedback"]: gr.update(visible=True)}


def hide_about_panel(accept_checkbox):
    # Stay open while not accepted
    open = not bool(accept_checkbox)
    print(open)
    return {about_panel: gr.update(open=open)}


def setup_feedback_form():
    # Feedback
    feedback_elems = {}
    with gr.Box():
        with gr.Row():
            with gr.Box():
                with gr.Column():
                    gr.Markdown(
                        f"""## Help Us Improve with Your Feedback! 👍👎
By filling out the feedback form, you're helping us understand what's working well and where we can make improvements on {app_name}.

Every thumbs up or thumbs down helps us determine how to improve {app_name}. Thank you for contributing to the evaluation of our app.

"""
                    )
                    with gr.Row():
                        overall_experience = gr.Radio(
                            choices=["👍", "👎"], label="Did you find what you were looking for?"
                        )

                    show_additional_feedback = gr.Group(visible=False)
                    with show_additional_feedback:
                        with gr.Column():
                            clear_answer = gr.Radio(
                                choices=["👍", "👎"], label="Was the generated answer clear and understandable?"
                            )
                            accurate_answer = gr.Radio(choices=["👍", "👎"], label="Was the generated answer accurate?")
                            safe_answer = gr.Radio(choices=["👍", "👎"], label="Was the generated answer safe?")
                            relevant_sources = gr.Radio(
                                choices=["👍", "👎"], label="Were the retrieved sources generally relevant to your query?"
                            )
                            relevant_sources_selection = gr.CheckboxGroup(
                                choices=[f"Source {i+1}" for i in range(max_sources)],
                                label="Check all relevant sources",
                            )

                            extra_info = gr.Textbox(
                                label="Enter additional information (optional)",
                                lines=3,
                                placeholder="Enter more helpful information for us here...",
                            )

                    submit_feedback_btn = gr.Button("Submit Feedback!")
                    with gr.Column(visible=False) as submitted_message:
                        gr.Markdown("Feedback recorded, thank you! 📝")

    overall_experience.input(toggle_additional_feedback, outputs=show_additional_feedback)

    # fmt: off
    submit_feedback_btn.click(
        toggle_feedback_visible,
        inputs=gr.State(False),
        outputs=submitted_message,
    ).then(
        submit_feedback,
        inputs=[
            overall_experience, clear_answer, accurate_answer, safe_answer, relevant_sources, relevant_sources_selection, extra_info, user_completions
        ],
    ).success(
        toggle_feedback_visible,
        inputs=gr.State(True),
        outputs=submitted_message,
    )
    # If you rage click the subimt feedback button, it re-appears so you are confident it was recorded properly.
    # fmt: on
    feedback_elems = {
        "overall_experience": overall_experience,
        "clear_answer": clear_answer,
        "accurate_answer": accurate_answer,
        "safe_answer": safe_answer,
        "relevant_sources": relevant_sources,
        "relevant_sources_selection": relevant_sources_selection,
        "submit_feedback_btn": submit_feedback_btn,
        "submitted_message": submitted_message,
        "show_additional_feedback": show_additional_feedback,
        "extra_info": extra_info,
    }

    return feedback_elems


def to_md_link(title: str, link: str) -> str:
    """Converts a title and link to markown link format"""
    return f"[{title}]({link})"


def get_metadata_markdown(df) -> str:
    """Converts the content from a dataframe to a markdown table string format."""
    metadata = []

    # Order articles by year, with latest first
    df = df.sort_values(["Country", "Year"], ascending=True)

    for _, item in df.iterrows():
        # source = item["Source"]
        link = item["Link"]
        title = item["Title"]
        year = item["Year"]
        country = item["Country"]

        metadata.append(f"{year} | {country} | {to_md_link(title, link)} ")
    metadata_str = "\n".join(metadata)

    markdown_text = f"""
| Year | Country | Report |
| ---    | --- | --- |
{metadata_str}
"""
    return markdown_text


def append_completion(completion, user_completions):
    user_completions.append(completion)
    return user_completions


def add_user_question(user_question: str, chat_history: Optional[ChatHistory] = None) -> ChatHistory:
    """Adds a user's question to the chat history.

    If no history is provided, the first element of the history will be the user conversation.
    """
    if chat_history is None:
        chat_history = []
    chat_history.append([user_question, None])
    return chat_history


def chat(chat_history: ChatHistory):
    """Answer a user's question using retrieval augmented generation."""

    # We assume that the question is the user's last interaction
    user_input = chat_history[-1][0]

    completion = buster.process_input(user_input)

    # Stream tokens one at a time
    chat_history[-1][1] = ""
    for token in completion.answer_generator:
        chat_history[-1][1] += token

        yield chat_history, completion


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
    overall_experience,
    clear_answer,
    accuracte_answer,
    safe_answer,
    relevant_sources,
    relevant_sources_selection,
    extra_info,
    user_completions,
    request: gr.Request,
):
    feedback_form = FeedbackForm(
        overall_experience=overall_experience,
        clear_answer=clear_answer,
        accurate_answer=accuracte_answer,
        safe_answer=safe_answer,
        relevant_sources=relevant_sources,
        relevant_sources_selection=relevant_sources_selection,
        extra_info=extra_info,
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
    return {feedback_elems["submitted_message"]: gr.update(visible=visible)}


def clear_sources():
    """Clears all the documents in the tabs"""
    return ["" for _ in range(max_sources)]


def clear_feedback_form():
    """Clears the contents of the feedback form."""
    return {
        feedback_elems["overall_experience"]: gr.update(value=None),
        feedback_elems["clear_answer"]: gr.update(value=None),
        feedback_elems["accurate_answer"]: gr.update(value=None),
        feedback_elems["safe_answer"]: gr.update(value=None),
        feedback_elems["relevant_sources"]: gr.update(value=None),
        feedback_elems["relevant_sources_selection"]: gr.update(value=None),
        feedback_elems["extra_info"]: gr.update(value=None),
    }


def reveal_app(checkbox: bool):
    if checkbox:
        return gr.Group.update(visible=True), gr.Group.update(visible=False)
    else:
        gr.Warning("You must accept terms and conditions to continue...")
        return gr.Group.update(visible=False), gr.Group.update(visible=True)


def display_sources():
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
    return sources_textboxes


buster_app = gr.Blocks()


def init_about_panel():
    with gr.Accordion(label="About Panel", open=True) as about_panel:
        with gr.Row(variant="panel"):
            with gr.Box():
                gr.Markdown(
                    f"""

                ## Welcome
                {app_name} is a question-answering chatbot on AI policies from various sources.
                Using this platform, you can ask AI-policy questions and {app_name} will look for the most relevant policies at its disposal to formulate an answer.
                All of its available sources are listed on the bottom of the page.

                ## How it works
                This app uses language models to convert documents to their semanatic representations.
                When a user asks a question, {app_name} compares it against all available documents at its disposal. It then retrieves the documents that are most relevant to your question and prompts ChatGPT with these documents to generate a response.
                The answer and accompanying sources are then displayed so that you can verify the veracity of the generated responses.
                """
                )

            with gr.Box():
                gr.Markdown(
                    f"""
                ## Limitations

                {app_name} is intended to ***_only be used as a demo._*** While we have worked hard to make this as useful as possible, it is important to understand that there are no guarantees regarding the accuracy of its responses.
                Like all language models, {app_name} might generate information that is not entirely reliable and sometimes hallucinate responses. To mitigate this, users are strongly advised to independently verify the information provided by the tool.
                All sources available to the model are listed below.

                ## Recommended usage

                For optimal results, employ {app_name} in scenarios where the answers to questions can be found concisely within the provided documentation.
                For questions that demand complex reasoning spanning across an entire document, multiple documents or require contextual understanding, the performance of {app_name} might be limited.

                When the model fails to find relevant information, it will advise a user that it cannot answer a question.
                The model is also instructed to ignore questions that are not directly related to AI policies.
                """
                )

    return about_panel


def setup_terms_and_conditions():
    terms_and_conditions = """App Terms and Conditions

    (NOTE: These are autogenerated by ChatGPT and is intended as a placeholder, DO NOT PUBLISH THESE)
    Welcome to our question answering bot app ("the App"). Your use of the App is subject to the following terms:

    1. Usage Agreement

    By using the App, you agree to these terms and any future changes.

    2. App Purpose

    The App provides question answering. It doesn't collect personal info. Interactions are stored for App improvement.

    3. Data Collection

    a. The App records and stores interactions without personal info for enhancement.

    b. Non-personal data like device type may be collected for optimization.

    4. Data Use

    a. Stored interactions improve App accuracy and functionality.

    b. Aggregated data is analyzed for user trends and better experience.

    5. Security

    Data security measures are in place to prevent unauthorized access.

    6. Changes

    Terms may change; continued use means acceptance of changes.

    7. Termination

    Access can be terminated for breach of terms.

    8. Governing Law

    Terms follow the laws of [Your Jurisdiction]. Disputes subject to [Your Jurisdiction]'s courts.

    9. Contact

    Questions? Reach us at [Contact Email].

    Using the App means you've read, understood, and agreed to these terms. Thanks for using our question answering bot App!
    """

    accept_terms_group = gr.Group(visible=enable_terms_and_conditions)
    with accept_terms_group:
        with gr.Column(variant="compact"):
            with gr.Box():
                gr.Markdown(
                    f"## Terms and Coniditions \n Welcome to {app_name} Before continuing, you must read and accept the terms and conditions."
                )
                gr.Textbox(terms_and_conditions, interactive=False, max_lines=10, label="Terms & Conditions")
                with gr.Column():
                    accept_checkbox = gr.Checkbox(label="I accept the terms.", interactive=True)
                    accept_terms = gr.Button("Enter", variant="primary")
    return accept_terms_group, accept_checkbox, accept_terms


with buster_app:
    # TODO: trigger a proper change to update

    # state variables are client-side and are reset every time a client refreshes the page
    user_completions = gr.State([])

    gr.Markdown(f"<h1><center>{app_name}: A Question-Answering Bot for your documentation</center></h1>")

    about_panel = init_about_panel()

    accept_terms_group, accept_checkbox, accept_terms = setup_terms_and_conditions()

    app_group_visible = False if enable_terms_and_conditions else True
    app_group = gr.Box(visible=app_group_visible)
    with app_group:
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
                sources_textboxes = display_sources()

            with gr.Column():
                gr.Markdown("## Example questions")
                gr.Examples(
                    examples=relevant_questions[0:5],
                    inputs=message,
                    label="Questions users could ask.",
                )

                feedback_elems = setup_feedback_form()

        # Display additional sources
        with gr.Box():
            with gr.Column():
                gr.Markdown(
                    f"""## 📚 Sources
                Here we list all of the sources that {app_name} has access to.
                """
                )
                # TODO: Pick how to display the sources, 2 options for now
                # Display the sources using a dataframe (rendering options limited)...
                # gr.DataFrame(documents_metadata, headers=list(documents_metadata.columns), interactive=False)

                # ... Or display the sources using markdown.
                gr.Markdown(get_metadata_markdown(documents_metadata))

        gr.HTML("<center> Powered by <a href='https://github.com/jerpint/buster'>Buster</a> 🤖</center>")

        # store the users' last completion here
        completion = gr.State()

    # fmt: off
    # Reval app if terms are accepted, once accepted
    accept_terms.click(
        reveal_app,
        inputs=accept_checkbox,
        outputs=[app_group, accept_terms_group]
    ).then(
        hide_about_panel,
        inputs=accept_checkbox,
        outputs=about_panel,
    )
    # fmt: on

    # fmt: off
    submit.click(
        add_user_question, [message], [chatbot]
    ).then(
        clear_sources,
        outputs=[*sources_textboxes]
    ).then(
        clear_feedback_form,
        outputs=[
            feedback_elems["overall_experience"],
            feedback_elems["clear_answer"],
            feedback_elems["accurate_answer"],
            feedback_elems["safe_answer"],
            feedback_elems["relevant_sources"],
            feedback_elems["relevant_sources_selection"],
            feedback_elems["extra_info"],
        ]
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
        add_user_question, [message], [chatbot]
    ).then(
        clear_sources,
        outputs=[*sources_textboxes]
    ).then(
        clear_feedback_form,
        outputs=[
            feedback_elems["overall_experience"],
            feedback_elems["clear_answer"],
            feedback_elems["accurate_answer"],
            feedback_elems["safe_answer"],
            feedback_elems["relevant_sources"],
            feedback_elems["relevant_sources_selection"],
            feedback_elems["extra_info"],
        ]
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
