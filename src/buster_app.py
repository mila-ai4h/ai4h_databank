import copy
import logging
import os
from pathlib import Path
from typing import Optional, Union

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

# sample questions
example_questions = [
    "Are there any AI policies related to AI adoption in the public sector in the UK?",
    "How is Canada evaluating the success of its AI strategy?",
    "Has the EU proposed specific legislation on AI?",
]

enable_terms_and_conditions = True

app_name = "LLaWma 🦙"


def hide_about_panel(accept_checkbox):
    # Stay open while not accepted
    open = not bool(accept_checkbox)
    return {about_panel: gr.update(open=open)}


def setup_feedback_form():
    # Feedback
    feedback_elems = {}
    with gr.Box():
        with gr.Row():
            with gr.Box():
                with gr.Column():
                    gr.Markdown(
                        f""" ## Thank You For Being Here!

Thank you for being here and providing feedback on the model's outputs! Your feedback will help us make the tool as useful as possible for the community!

Since this tool is still in its early stages of development, please only engage with it as a demo and not for use in a proper research context.

We look forward to sharing with you an updated version of the product once we feel it's ready!
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
                            relevant_sources_order = gr.Radio(
                                choices=["👍", "👎"], label="Were the sources ranked appropriately, in order of relevance?"
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

                    submit_feedback_btn = gr.Button("Submit Feedback!", variant="primary", interactive=False)
                    with gr.Column(visible=False) as submitted_message:
                        gr.Markdown("Feedback recorded, thank you! 📝")

    overall_experience.input(toggle_visibility, inputs=gr.State("True"), outputs=show_additional_feedback)

    # fmt: off
    submit_feedback_btn.click(
        toggle_visibility,
        inputs=gr.State(False),
        outputs=submitted_message,
    ).then(
        submit_feedback,
        inputs=[
            overall_experience, clear_answer, accurate_answer, safe_answer, relevant_sources, relevant_sources_order, relevant_sources_selection, extra_info, last_completion,
        ],
    ).success(
        toggle_visibility,
        inputs=gr.State(True),
        outputs=submitted_message,
    ).success(
        toggle_interactivity,
        inputs=gr.State(False),
        outputs=submit_feedback_btn,
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
    completion: Union[Completion, list[Completion]],
    request: gr.Request,
):

    # TODO: add UID for each page visitor instead of username

    # Get the proper mongo collection to save logs to
    collection = cfg.mongo_interaction_collection

    if isinstance(completion, Completion):
        user_completions = [completion]

    interaction = Interaction(
        user_completions=user_completions,
        time=get_utc_time(),
        username=request.username,
    )
    interaction.send(mongo_db, collection=collection)


def submit_feedback(
    overall_experience: str,
    clear_answer: str,
    accuracte_answer: str,
    safe_answer: str,
    relevant_sources: str,
    relevant_sources_order: list[str],
    relevant_sources_selection: str,
    extra_info: str,
    completion: Union[Completion, list[Completion]],
    request: gr.Request,
):
    feedback_form = FeedbackForm(
        overall_experience=overall_experience,
        clear_answer=clear_answer,
        accurate_answer=accuracte_answer,
        safe_answer=safe_answer,
        relevant_sources=relevant_sources,
        relevant_sources_order=relevant_sources_order,
        relevant_sources_selection=relevant_sources_selection,
        extra_info=extra_info,
    )

    if isinstance(completion, Completion):
        user_completions = [completion]

    feedback = Interaction(
        user_completions=user_completions,
        form=feedback_form,
        time=get_utc_time(),
        username=request.username,
    )
    feedback.send(mongo_db, collection=cfg.mongo_feedback_collection)


def toggle_visibility(visible: bool):
    """Toggles the visibility of the 'feedback submitted' message."""
    return gr.update(visible=visible)


def toggle_interactivity(interactive: bool):
    """Toggles the visibility of the 'feedback submitted' message."""
    return gr.update(interactive=interactive)


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


def setup_about_panel():
    with gr.Accordion(label=f"About {app_name}", open=True) as about_panel:
        with gr.Row(variant="panel"):
            with gr.Box():
                gr.Markdown(f"""

                ## Welcome
                Artificial intelligence is a field that's developing fast! In response, policy makers from around the world are creating guidelines, rules and regulations to keep up.

                Finding accurate and up-to-date information about regulatory changes can be difficult but crucial to share best practices, ensure interoperability and promote adherence to local laws and regulations. That's why we've created {app_name}.

                {app_name} is a Q&A chatbot designed to provide relevant and high quality information about AI policies from around the world. Using this tool, your AI policy questions will be answered, accompanied by relevant analyses by the OECD's AI Observatory!

                ## How it works (and doesn't)

                {app_name} uses Large Language Models (AI algorithms that work with text) to pinpoint sections of policy documents that are relevant to your question. Rather than presenting you with the specific policy section verbatim, {app_name} has been designed to summarize the information in a digestible format, so that the response you receive more naturally fits with the question you've posed.
                """
                )

            with gr.Box():
                gr.Markdown(f"""
                ## Risks

                We have done our best to make sure that the AI algorithms are __only__ taking information from what is available in the OECD AI Observatory’s Database; but, of course, Large Language Models (LLMs) are prone to fabrication. This means LLMs can make things up and present this made up information as if it were real, making it seem as if the information was found in a policy document. We therefore advise you to check the sources provided by the model to validate that the answer is in fact true. If you'd like to know exactly which documents the model can reference in its response, please see below.


                ## Recommended usage

                {app_name} can only answer specific types of questions, for example:

                * Questions about policy documents that are currently in the OECD AI Observatory's database
                * Questions that are posed in English and target English language documents;
                * Questions for which the answer can be found in the text (i.e. the thinking has already been done by the author) these AI models are not able to write their own research report combining information across policy documents and analyzing them itself).

                If your question is outside the scope of the recommended use, the model has been instructed not to answer.
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
    # state variables are client-side and are reset every time a client refreshes the page
    # store the users' last completion here
    last_completion = gr.State()

    gr.Markdown(f"<h1><center>{app_name}: A Question-Answering Bot for your documentation</center></h1>")

    about_panel = setup_about_panel()

    accept_terms_group, accept_checkbox, accept_terms = setup_terms_and_conditions()

    app_group_visible = False if enable_terms_and_conditions else True
    app_group = gr.Box(visible=app_group_visible)
    with app_group:
        with gr.Row():
            with gr.Column(scale=2, variant="panel"):
                gr.Markdown("## Chatbot")
                chatbot = gr.Chatbot(label=f"{app_name}")
                message = gr.Textbox(
                    label=f"Chat with {app_name}",
                    placeholder="Ask your question here...",
                    lines=1,
                )
                submit = gr.Button(value="Send", variant="primary")
                sources_textboxes = display_sources()

            with gr.Column():
                gr.Markdown("## Example questions")
                gr.Examples(
                    examples=example_questions,
                    inputs=message,
                    label="Questions users could ask.",
                )

                feedback_elems = setup_feedback_form()

        # Display additional sources
        with gr.Box():
            gr.Markdown(f"")

            gr.Markdown(
                f"""## 📚 Sources
            {app_name} has access to dozens of AI policy documents from various sources.
            Below we list all of the sources that {app_name} has access to.
            """
            )
            with gr.Accordion(open=False, label="Click to list all available sources 📚"):
                with gr.Column():
                    # TODO: Pick how to display the sources, 2 options for now
                    # Display the sources using a dataframe (rendering options limited)...
                    # gr.DataFrame(documents_metadata, headers=list(documents_metadata.columns), interactive=False)

                    # ... Or display the sources using markdown.
                    gr.Markdown(get_metadata_markdown(documents_metadata))

        gr.HTML("<center> Powered by <a href='https://github.com/jerpint/buster'>Buster</a> 🤖</center>")


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
        toggle_interactivity,
        inputs=gr.State("True"),
        outputs=feedback_elems["submit_feedback_btn"],
    ).then(
        chat,
        inputs=[chatbot],
        outputs=[chatbot, last_completion],
    ).then(
        add_sources,
        inputs=[last_completion, gr.State(max_sources)],
        outputs=[*sources_textboxes]
    ).then(
        log_completion,
        inputs=last_completion,
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
        outputs=[chatbot, last_completion],
    ).then(
        add_sources,
        inputs=[last_completion, gr.State(max_sources)],
        outputs=[*sources_textboxes]
    ).then(
        log_completion,
        inputs=last_completion,
    )
    # fmt: on


# True when launching using gradio entrypoint
if os.getenv("MOUNT_GRADIO_APP") is None:
    logger.info("launching app via gradio")
    buster_app.queue(concurrency_count=16)
    buster_app.launch(share=False, auth=check_auth)
