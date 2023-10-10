import copy
import logging
from typing import Optional, Union

import gradio as gr
import pandas as pd

import src.cfg as cfg
from buster.completers import Completion
from src.app_utils import add_sources, get_session_id, get_utc_time
from src.cfg import setup_buster
from src.feedback import FeedbackForm, Interaction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Typehint for chatbot history
ChatHistory = list[list[Optional[str], Optional[str]]]

app_name = cfg.app_name
example_questions = cfg.example_questions
disclaimer = cfg.disclaimer
mongo_db = cfg.mongo_db
buster_cfg = copy.deepcopy(cfg.buster_cfg)
buster = setup_buster(buster_cfg=buster_cfg)
max_sources = cfg.buster_cfg.retriever_cfg["top_k"]
data_dir = cfg.data_dir

# get documents metadata
documents_metadata_file = str(data_dir / "documents_metadata.csv")
documents_metadata = pd.read_csv(documents_metadata_file)

enable_terms_and_conditions = False


def add_disclaimer(completion: Completion, chat_history: ChatHistory, disclaimer: str = disclaimer):
    """Add a disclaimer response if the answer was relevant."""
    if completion.answer_relevant:
        chat_history.append([None, disclaimer])
    return chat_history


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
                            relevant_sources = gr.Radio(
                                choices=["👍", "👎"], label="Were the retrieved sources generally relevant to your query?"
                            )
                            relevant_sources_selection = gr.CheckboxGroup(
                                choices=[f"Source {i+1}" for i in range(max_sources)],
                                label="Check all relevant sources (if any)",
                            )
                            relevant_sources_order = gr.Radio(
                                choices=["👍", "👎"],
                                label="Were the sources ranked appropriately, in order of relevance?",
                            )

                            extra_info = gr.Textbox(
                                label="Enter any additional information",
                                lines=3,
                                placeholder="Enter more helpful information for us here...",
                            )

                    submit_feedback_btn = gr.Button("Submit Feedback!", variant="primary", interactive=False)
                    with gr.Column(visible=False) as submitted_message:
                        gr.Markdown("Feedback recorded, thank you! 📝")

    # fmt: off
    overall_experience.input(
        toggle_visibility,
        inputs=gr.State("True"),
        outputs=show_additional_feedback
    ).then(
        toggle_interactivity,
        inputs=gr.State(True),
        outputs=submit_feedback_btn,
    )
    # fmt: on

    # fmt: off
    submit_feedback_btn.click(
        toggle_visibility,
        inputs=gr.State(False),
        outputs=submitted_message,
    ).then(
        submit_feedback,
        inputs=[
            overall_experience, clear_answer, accurate_answer, relevant_sources, relevant_sources_order, relevant_sources_selection, extra_info, last_completion, session_id,
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

    # fmt: on
    feedback_elems = {
        "overall_experience": overall_experience,
        "clear_answer": clear_answer,
        "accurate_answer": accurate_answer,
        "relevant_sources": relevant_sources,
        "relevant_sources_selection": relevant_sources_selection,
        "relevant_sources_order": relevant_sources_order,
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
    collection: str,
    session_id: str,
    request: gr.Request,
    instance_type: Optional[str] = cfg.INSTANCE_TYPE,
    instance_name: Optional[str] = cfg.INSTANCE_NAME,
    mongo_db=cfg.mongo_db,
):
    """
    Log user completions in a specified collection for analytics.

    Parameters:
    completion (Union[Completion, list[Completion]]): A single completion or a list of completions
        to log. Completions can be instances of the Completion class.
    collection (str): The name of the MongoDB collection where the interactions will be stored.
    session_id (str): A unique identifier for the current session. In gradio this is reset every time a page is refreshed.
    request (gr.Request): The gradio request object containing request metadata.
    instance_type (str, optional): The type of instance where the completion took place.
        Defaults to cfg.INSTANCE_TYPE.
    instance_name (str, optional): The name of the instance where the completion took place.
        Defaults to cfg.INSTANCE_NAME.
    """

    # TODO: add UID for each page visitor instead of username

    # make sure it's always a list
    if isinstance(completion, Completion):
        user_completions = [completion]
    else:
        user_completions = completion

    interaction = Interaction(
        user_completions=user_completions,
        time=get_utc_time(),
        username=request.username,
        session_id=session_id,
        instance_name=instance_name,
        instance_type=instance_type,
    )
    interaction.send(mongo_db, collection=collection)


def submit_feedback(
    overall_experience: str,
    clear_answer: str,
    accuracte_answer: str,
    relevant_sources: str,
    relevant_sources_order: list[str],
    relevant_sources_selection: str,
    extra_info: str,
    completion: Union[Completion, list[Completion]],
    session_id: str,
    request: gr.Request,
    instance_type: Optional[str] = cfg.INSTANCE_TYPE,
    instance_name: Optional[str] = cfg.INSTANCE_NAME,
):
    feedback_form = FeedbackForm(
        overall_experience=overall_experience,
        clear_answer=clear_answer,
        accurate_answer=accuracte_answer,
        relevant_sources=relevant_sources,
        relevant_sources_order=relevant_sources_order,
        relevant_sources_selection=relevant_sources_selection,
        extra_info=extra_info,
    )

    # make sure it's always a list
    if isinstance(completion, Completion):
        user_completions = [completion]
    else:
        user_completions = completion

    feedback = Interaction(
        user_completions=user_completions,
        form=feedback_form,
        time=get_utc_time(),
        username=request.username,
        session_id=session_id,
        instance_name=instance_name,
        instance_type=instance_type,
    )
    feedback.send(mongo_db, collection=cfg.MONGO_COLLECTION_FEEDBACK)


def toggle_visibility(visible: bool):
    """Toggles the visibility of the gradio element."""
    return gr.update(visible=visible)


def toggle_interactivity(interactive: bool):
    """Toggles the visibility of the gradio element."""
    return gr.update(interactive=interactive)


def clear_user_input():
    """Clears the contents of the user_input box."""
    return gr.update(value="")


def clear_sources():
    """Clears all the documents in the tabs"""
    return ["" for _ in range(max_sources)]


def clear_feedback_form():
    """Clears the contents of the feedback form."""
    return {
        feedback_elems["overall_experience"]: gr.update(value=None),
        feedback_elems["clear_answer"]: gr.update(value=None),
        feedback_elems["accurate_answer"]: gr.update(value=None),
        feedback_elems["relevant_sources"]: gr.update(value=None),
        feedback_elems["relevant_sources_selection"]: gr.update(value=None),
        feedback_elems["relevant_sources_order"]: gr.update(value=None),
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
                t = gr.Markdown(latex_delimiters=[])
            sources_textboxes.append(t)
    return sources_textboxes


buster_app = gr.Blocks()


def setup_about_panel():
    with gr.Accordion(label=f"About {app_name}", open=False) as about_panel:
        with gr.Row(variant="panel"):
            with gr.Box():
                gr.Markdown(
                    f"""

                ## Welcome
                Artificial intelligence is a field that's developing fast! In response, policy makers from around the world are creating guidelines, rules and regulations to keep up.

                Finding accurate and up-to-date information about regulatory changes can be difficult but crucial to share best practices, ensure interoperability and promote adherence to local laws and regulations. That's why we've created {app_name}.

                {app_name} is a Q&A search engine designed to provide relevant and high quality information about AI policies from around the world. Using this tool, your AI policy questions will be answered, accompanied by relevant analyses by the OECD's AI Observatory!

                ## How it works (and doesn't)

                {app_name} uses Large Language Models (AI algorithms that work with text) to pinpoint sections of policy documents that are relevant to your question. Rather than presenting you with the specific policy section verbatim, {app_name} has been designed to summarize the information in a digestible format, so that the response you receive more naturally fits with the question you've posed.
                """
                )

            with gr.Box():
                gr.Markdown(
                    f"""
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
    terms_and_conditions = f"""App Terms and Conditions

-	Usage Agreement: By using the {app_name}, you agree to its terms and conditions. Terms may change; continued use means acceptance of changes.

-	Background: Your use of the {app_name} is subject to the terms and conditions found at www.oecd.org/termsandconditions. The following disclosures do not modify or supersede those terms. Instead, these disclosures aim to provide greater transparency surrounding information included in the {app_name}.

-	Third-Party Information: The {app_name} serves as an accessible starting point for comprehending the AI policy landscape. As a result, please be aware that the {app_name} may retrieve information from documents, articles and data from various third-parties with which the OECD may not have an affiliation.

-	Data collection: The {app_name} may record and store user interactions with the purpose of improving the model and its outputs. It does not collect personal data. Please do not include personally identifiable information (PII) in your queries.

-	Views Expressed: Please understand that any views or opinions expressed on the {app_name} are solely those of the third-parties that developed or collected the training data and do not represent the views or opinions of the OECD. Further, the inclusion of any document or dataset does not constitute an endorsement or recommendation by the OECD.

-	Use of generative AI: The {app_name} uses generative AI technologies to process and analyse data and information and to respond to user queries.

-	Errors and Omissions: The OECD cannot guarantee and does not independently verify the accuracy, completeness, or validity information provided in the {app_name}. You should be aware that information included in the {app_name} may contain various errors and omissions and should be treated accordingly. Ensuring the veracity and accuracy of the information provided by the {app_name} is the User’s responsibility.

-	Scope: The {app_name} is not designed to write its own research by combining information across policy documents, analysing them and extracting insights. The {app_name} is not designed to answer questions for which no relevant information can be found in the training data.

-	Intellectual Property: Any of the copyrights, trademarks, service marks, collective marks, design rights, or other intellectual property or proprietary rights that are mentioned, cited, or otherwise included in the {app_name} are the property of their respective owners. Their use or inclusion in the {app_name} does not imply that you may use them for any other purpose. The OECD is not endorsed by, does not endorse, and may not affiliated with any of the holders of such rights, and as such, the OECD cannot and do not grant any rights to use or otherwise exploit the protected materials included herein.

-	Limitation of Liability: Under no circumstances shall OECD be liable to any user on account of that user’s use or misuse or reliance on the {app_name}.

-	Termination: OECD reserves the right to limit or suspend access by the users to {app_name} at any time and without notice for use deemed to be a breach to the terms and conditions.

-	Timeliness: The AI policy landscape is rapidly evolving. As such, while we regularly update our database, some of the information might become outdated or may not reflect the most recent changes or additions to policies and regulations.

-	Contact: Questions? Reach us at ai@oecd.org.


    """

    accept_terms_group = gr.Group(visible=enable_terms_and_conditions)
    with accept_terms_group:
        with gr.Column(variant="compact"):
            with gr.Box():
                gr.Markdown(
                    f"## Terms and Conditions \n Welcome to {app_name} Before continuing, you must read and accept the terms and conditions."
                )
                gr.Textbox(terms_and_conditions, interactive=False, max_lines=10, label="Terms & Conditions")
                with gr.Column():
                    accept_checkbox = gr.Checkbox(label="I accept the terms.", interactive=True)
                    accept_terms = gr.Button("Enter", variant="primary")
    return accept_terms_group, accept_checkbox, accept_terms


def setup_additional_sources():
    # Display additional sources
    with gr.Box():
        gr.Markdown(f"")

        gr.Markdown(
            f"""## 📚 Sources
        {app_name} has access to dozens of AI policy documents from various sources.
        Below we list all of the sources that {app_name} has access to.
        """
        )
        with gr.Accordion(open=True, label="Click to list all available sources 📚"):
            with gr.Column():
                # Display the sources using a dataframe table
                documents_metadata["Report"] = documents_metadata.apply(
                    lambda row: to_md_link(row["Title"], row["Link"]), axis=1
                )
                sub_df = documents_metadata[["Country", "Year", "Report"]]
                gr.DataFrame(
                    sub_df, headers=list(sub_df.columns), interactive=False, datatype=["number", "str", "markdown"]
                )

                # Uncomment to display the sources instead as a simple markdown table
                # gr.Markdown(get_metadata_markdown(documents_metadata))


def raise_flagging_message():
    """Raises a red banner indicating that the content has been flagged."""
    raise gr.Error(
        "Thank you for flagging the content. Our moderation team will look closely at these samples. We appologize for any harm this might have caused you."
    )


def setup_flag_button():
    """Sets up a flag button with some accompanying text explaining why we have it."""
    with gr.Column(variant="compact"):
        with gr.Box():
            gr.Markdown(
                """# Report Misuse
    While we took many steps to ensure the tool is safe, we still rely on third parties for our LLM capabilities. Please let us know if any harmful content shows up by clicking the button below. You can also send us screenshots/concerns to mila.databank@gmail.com"""
            )
            flag_button = gr.Button(value="Flag Content 🚩")
    return flag_button


with buster_app:
    # State variables are client-side and are reset every time a client refreshes the page
    # Store the users' last completion here
    last_completion = gr.State()

    # A unique identifier that resets every time a page is refreshed
    session_id = gr.State(get_session_id)

    gr.Markdown(f"<h1><center>{app_name}: A Question-Answering Bot on AI Policies </center></h1>")

    about_panel = setup_about_panel()

    accept_terms_group, accept_checkbox, accept_terms = setup_terms_and_conditions()

    app_group_visible = False if enable_terms_and_conditions else True
    app_group = gr.Box(visible=app_group_visible)
    with app_group:
        with gr.Row():
            with gr.Column(scale=2, variant="panel"):
                gr.Markdown(
                    f"""
                Ask {app_name} your AI policy questions! Keep in mind this tool is a demo and can sometimes provide inaccurate information. Always verify the integrity of the information using the provided sources.
                """
                )
                with gr.Row():
                    with gr.Column(scale=10):
                        user_input = gr.Textbox(
                            label="",
                            placeholder=f"Ask {app_name}",
                            lines=1,
                        )
                    submit = gr.Button(value="Ask", variant="primary", size="lg")

                gr.Markdown(
                """
                By using this tool you agree to our [terms and conditions](file=src/buster/assets/index.html)
                """
                )
                chatbot = gr.Chatbot(label="Demo")
                sources_textboxes = display_sources()

            with gr.Column():
                gr.Markdown("## Example questions")
                gr.Examples(
                    examples=example_questions,
                    inputs=user_input,
                    label="Questions users could ask.",
                )

                feedback_elems = setup_feedback_form()

                flag_button = setup_flag_button()

        setup_additional_sources()

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
        add_user_question, [user_input], [chatbot]
    ).then(
        clear_user_input,
        outputs=[user_input]
    ).then(
        clear_sources,
        outputs=[*sources_textboxes]
    ).then(
        toggle_visibility,
        inputs=gr.State(False),
        outputs=feedback_elems["submitted_message"],
    ).then(
        toggle_visibility,
        inputs=gr.State(False),
        outputs=feedback_elems["show_additional_feedback"],
    ).then(
      clear_feedback_form,
        outputs=[
            feedback_elems["overall_experience"],
            feedback_elems["clear_answer"],
            feedback_elems["accurate_answer"],
            feedback_elems["relevant_sources"],
            feedback_elems["relevant_sources_selection"],
            feedback_elems["relevant_sources_order"],
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
        inputs=[last_completion, gr.State(cfg.MONGO_COLLECTION_INTERACTION), session_id]
    )

    user_input.submit(
        add_user_question, [user_input], [chatbot]
    ).then(
        clear_user_input,
        outputs=[user_input]
    ).then(
        clear_sources,
        outputs=[*sources_textboxes]
    ).then(
        toggle_visibility,
        inputs=gr.State(False),
        outputs=feedback_elems["submitted_message"],
    ).then(
        toggle_visibility,
        inputs=gr.State(False),
        outputs=feedback_elems["show_additional_feedback"],
    ).then(
      clear_feedback_form,
        outputs=[
            feedback_elems["overall_experience"],
            feedback_elems["clear_answer"],
            feedback_elems["accurate_answer"],
            feedback_elems["relevant_sources"],
            feedback_elems["relevant_sources_selection"],
            feedback_elems["relevant_sources_order"],
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
        inputs=[
            last_completion,
            gr.State(cfg.MONGO_COLLECTION_INTERACTION),
            session_id,
        ]
    )

    flag_button.click(
        log_completion,
        inputs=[last_completion, gr.State(cfg.MONGO_COLLECTION_FLAGGED), session_id]
    ).then(
        raise_flagging_message,
    )

    # fmt: on
