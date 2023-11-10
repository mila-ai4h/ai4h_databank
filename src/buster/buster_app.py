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
max_sources = cfg.max_sources
data_dir = cfg.data_dir


# link to the terms and conditions to be rendered in markdown blocks
path_to_tncs = "file=src/buster/assets/index.html"
md_link_to_tncs = f"[terms and conditions]({path_to_tncs})"

# get documents metadata
documents_metadata_file = str(data_dir / "documents_metadata.csv")
documents_metadata = pd.read_csv(documents_metadata_file)

css = """
.source {
    max-height: 250px; /* Set the maximum height for the textboxes */
    overflow: auto; /* Enable scrollbars when content exceeds dimensions */
    outline: 1px solid gray; /* Add a gray outline */
    border-radius: 5px; /* Add rounded corners to the outline */
}
"""


def add_disclaimer(completion: Completion, chat_history: ChatHistory, disclaimer: str = disclaimer):
    """Add a disclaimer response if the answer was relevant."""
    if completion.question_relevant:
        chat_history.append([None, disclaimer])
    return chat_history


def hide_about_panel(accept_checkbox):
    # Stay open while not accepted
    open = not bool(accept_checkbox)
    return {about_panel: gr.update(open=open)}


def set_relevant_sources_selection(num_sources: int):
    relevant_sources_selection = gr.CheckboxGroup(
        choices=[f"Source {i+1}" for i in range(num_sources)],
        label="Check all relevant sources (if any)",
    )
    return relevant_sources_selection


def setup_feedback_form(num_sources: int):
    # Feedback
    feedback_elems = {}
    with gr.Box():
        with gr.Row():
            with gr.Box():
                with gr.Column():
                    gr.Markdown(
                        f""" ## We would love your feedback!
Please submit feedback for each question asked.

Your feedback is anonymous and will help us make the tool as useful as possible for the community!
"""
                    )
                    with gr.Row():
                        overall_experience = gr.Radio(
                            choices=["👍", "👎"], label=f"Did {app_name} help answer your question?"
                        )

                    # Currently, we show all feedback, but also support having a small portion of it display at first
                    show_additional_feedback = gr.Group(visible=True)
                    with show_additional_feedback:
                        with gr.Column():
                            clear_answer = gr.Radio(
                                choices=["👍", "👎"], label="Was the generated answer clear and understandable?"
                            )
                            accurate_answer = gr.Radio(choices=["👍", "👎"], label="Was the generated answer accurate?")
                            relevant_sources = gr.Radio(
                                choices=["👍", "👎"], label="Were the retrieved sources generally relevant to your query?"
                            )
                            relevant_sources_selection = set_relevant_sources_selection(num_sources=num_sources)
                            relevant_sources_order = gr.Radio(
                                choices=["👍", "👎"],
                                label="Were the sources ranked appropriately, in order of relevance?",
                            )

                            extra_info = gr.Textbox(
                                label="Any other comments?",
                                lines=3,
                                placeholder="Please enter other feedback for improvement here...",
                            )

                            expertise = gr.Radio(
                                choices=["Beginner", "Intermediate", "Expert"],
                                label="How would you rate your knowledge of AI policy",
                                interactive=True,
                            )

                    submit_feedback_btn = gr.Button("Submit feedback", variant="primary", interactive=True)
                    with gr.Column(visible=False) as submitted_message:
                        gr.Markdown("Feedback recorded, thank you 📝! You can now ask a new question in the search bar.")

    # fmt: off
    submit_feedback_btn.click(
        toggle_visibility,
        inputs=gr.State(False),
        outputs=submitted_message,
    ).then(
        submit_feedback,
        inputs=[
            overall_experience,
            clear_answer,
            accurate_answer,
            relevant_sources,
            relevant_sources_order,
            relevant_sources_selection,
            expertise,
            extra_info,
            last_completion,
            session_id,
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
        "expertise": expertise,
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


def chat(chat_history: ChatHistory, reformulate_question: bool, top_k: Optional[int] = None):
    """Answer a user's question using retrieval augmented generation."""

    # Make sure top k is an int between 1 and 15
    top_k = int(top_k)
    top_k = max(top_k, 1)
    top_k = min(top_k, max_sources)

    # We assume that the question is the user's last interaction
    user_input = chat_history[-1][0]

    completion = buster.process_input(user_input, reformulate_question=reformulate_question, top_k=top_k)

    # ## FOR DEBUGGING ##
    # from buster.completers import UserInputs
    # completion = Completion(
    #     user_inputs=UserInputs(original_input=user_input, reformulated_input="New question"),
    #     error=False,
    #     matched_documents=None,
    #     answer_generator="Debugging some code right now..."
    # )
    # ##

    if reformulate_question and not completion.error:
        assert completion.user_inputs.reformulated_input is not None
        chat_history.append([None, f"I reformulated your question to: {completion.user_inputs.reformulated_input}"])
        chat_history.append([None, None])

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
        data_version=cfg.MONGO_DATABASE_DATA,
    )
    interaction.send(mongo_db, collection=collection)


def submit_feedback(
    overall_experience: str,
    clear_answer: str,
    accuracte_answer: str,
    relevant_sources: str,
    relevant_sources_order: list[str],
    relevant_sources_selection: str,
    expertise: list[str],
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
        expertise=expertise,
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
        feedback_elems["expertise"]: gr.update(value=None),
        feedback_elems["extra_info"]: gr.update(value=None),
    }


def reveal_app(choice: gr.SelectData):
    return (
        gr.Group.update(visible=False),
        gr.update(interactive=True, placeholder="Ask your AI policy question here…"),
        gr.update(interactive=True),
    )


def display_sources():
    with gr.Column():
        gr.Markdown(
            """## Relevant sources
        All retrieved documents will be listed here in order of importance.
        """
        )
        sources_textboxes = []
        for i in range(max_sources):
            t = gr.Markdown(latex_delimiters=[], elem_classes="source", visible=False)
            sources_textboxes.append(t)
    return sources_textboxes


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

                It's helpful to keep in mind that {app_name} is entirely restricted to our database (see “Available Sources” below). These sources are from the [OECD.AI](http://oecd.ai/) Database (containing national AI policies) and AI-related reports from the OECD iLibrary. If the answer to your question is not contained in these policy documents, the model won't be able to respond.

                Since we restrict the model to information found in the documentation, it has a hard time with questions that require more generalized knowledge. Therefore, if you ask the model for information about AI policies in Asia, the model won't necessarily show you Japanese policy documentation. To overcome this limitation, it's best to be as specific as possible in your question, referencing the particular country you're looking for information on.

                For more information about the tool's strengths and limitations, please see our website [here](https://mila.quebec/en/project/sai/).
                """
                )

            with gr.Box():
                gr.Markdown(
                    f"""
                ## Risks

                We have done our best to make sure that the AI algorithms are __only__ taking information from what is available in the OECD AI Observatory's Database; but, of course, Large Language Models (LLMs) are prone to fabrication. This means LLMs can make things up and present this made up information as if it were real, making it seem as if the information was found in a policy document. We therefore advise you to check the sources provided by the model to validate that the answer is in fact true. If you'd like to know exactly which documents the model can reference in its response, please see below.


                ## Recommended usage

                {app_name} can only answer specific types of questions, for example:

                * Questions about policy documents that are currently in the OECD AI Observatory's database
                * Questions that are posed in English and target English language documents;
                * Questions for which the answer can be found in the text (i.e. the thinking has already been done by the author) these AI models are not able to write their own research report combining information across policy documents and analyzing them itself).

                If your question is outside the scope of the recommended use, the model has been instructed not to answer.

                We are looking to create a tool that is as inclusive as possible.
                While currently the tool only works with English language questions and documents we will continue assessing {app_name}'s capacity to perform as intended for users with different levels of fluency in English and plan to expand the functionality to ensure accessibility and impact across countries and user groups.
                """
                )

    return about_panel


def setup_terms_and_conditions():
    with gr.Group(visible=True) as accept_terms_group:
        with gr.Column(scale=1):
            gr.Markdown(
                f"""
            By using this tool you agree to our {md_link_to_tncs}
            """,
            )
            accept_checkbox = gr.Checkbox(value=0, label="I accept", interactive=True, container=False, scale=1)
    return accept_terms_group, accept_checkbox


def setup_additional_sources():
    # Display additional sources
    with gr.Box():
        gr.Markdown(f"")

        gr.Markdown(
            f"""## 📚 Available sources
        {app_name} has access to dozens of AI policy documents from various sources.
        Below we list all of the sources that {app_name} has access to.
        """
        )
        with gr.Accordion(open=False, label="Click to list all available sources 📚"):
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
    gr.Info(
        "Thank you for flagging the content. Our moderation team will look closely at these samples. We appologize for any harm this might have caused you."
    )


def setup_flag_button():
    """Sets up a flag button with some accompanying text explaining why we have it."""
    with gr.Column(variant="compact"):
        with gr.Box():
            gr.Markdown(
                """# Report bugs and harmful content
    While we took many steps to ensure the tool is safe, we still rely on third parties for some of the model's capabilities. Please let us know if any harmful content shows up by clicking the button below and sending screenshots/concerns to mila.databank@gmail.com"""
            )
            flag_button = gr.Button(value="Flag content 🚩")
    return flag_button


def setup_user_settings():
    with gr.Accordion(label=f"Settings ⚙️", open=False):
        top_k_slider = gr.Slider(
            minimum=1,
            maximum=max_sources,
            interactive=True,
            value=5,
            step=1,
            label="Number of sources",
            info="Number of documents to pass to the language model during its retrieval.",
        )

        reformulate_question_cbox = gr.Checkbox(
            value=True,
            label="Reformulate Question (Beta)",
            info="Reformulates a user's question to enhance source retrieval.",
        )

    settings_elems = {
        "reformulate_question_cbox": reformulate_question_cbox,
        "top_k_slider": top_k_slider,
    }
    return settings_elems


buster_app = gr.Blocks(css=css)
with buster_app:
    # State variables are client-side and are reset every time a client refreshes the page
    # Store the users' last completion here
    last_completion = gr.State()

    # A unique identifier that resets every time a page is refreshed
    session_id = gr.State(get_session_id)

    gr.Markdown(f"<h1><center>SAI: Search engine for AI policy</center></h1>")

    about_panel = setup_about_panel()

    with gr.Row():
        with gr.Column(scale=2, variant="panel"):
            gr.Markdown(
                f"""
            Ask {app_name} your AI policy questions! Keep in mind this tool is a demo and can sometimes provide inaccurate information. Always verify the integrity of the information using the provided sources.
            Since this tool is still in its early stages of development, please only engage with it as a demo.
            """
            )
            accept_terms_group, accept_terms_checkbox = setup_terms_and_conditions()
            with gr.Row():
                with gr.Column(scale=20):
                    user_input = gr.Textbox(
                        label="",
                        placeholder="⚠️ Accept the terms and conditions to use the app",
                        lines=1,
                        interactive=False,
                    )
                submit = gr.Button(value="Ask", variant="primary", size="lg", interactive=False)

            gr.Examples(
                examples=example_questions,
                inputs=user_input,
                label=f"Sample questions to ask {app_name}",
            )

            chatbot = gr.Chatbot(label="Generated Answer", show_share_button=False)
            sources_textboxes = display_sources()

        with gr.Column():
            settings_elems = setup_user_settings()
            top_k_slider = settings_elems["top_k_slider"]
            feedback_elems = setup_feedback_form(top_k_slider.value)
            flag_button = setup_flag_button()

    top_k_slider.change(
        set_relevant_sources_selection, inputs=top_k_slider, outputs=feedback_elems["relevant_sources_selection"]
    )

    setup_additional_sources()

    gr.HTML(
        f"""
    <center>
        <div style='margin-bottom: 20px;'>  <!-- Add margin to the bottom of this div -->
            Powered by <a href='https://github.com/jerpint/buster'>Buster</a> 🤖
        </div>

        <div>
            <a href='{path_to_tncs}'> Terms And Conditions </a>
        </div>
    </center>
    """
    )

    # fmt: off
    # Allow use of submit button and hide checkbox when accepted
    accept_terms_checkbox.select(
        reveal_app,
        outputs=[accept_terms_group, user_input, submit]
    )

    gr.on(
        triggers=[submit.click, user_input.submit],
        fn=add_user_question,
        inputs=[user_input],
        outputs=[chatbot]
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
        toggle_interactivity,
        inputs=gr.State(True),
        outputs=feedback_elems["submit_feedback_btn"],
    ).then(
      clear_feedback_form,
        outputs=[
            feedback_elems["overall_experience"],
            feedback_elems["clear_answer"],
            feedback_elems["accurate_answer"],
            feedback_elems["relevant_sources"],
            feedback_elems["relevant_sources_selection"],
            feedback_elems["relevant_sources_order"],
            feedback_elems["expertise"],
            feedback_elems["extra_info"],
        ]
    ).then(
        chat,
        inputs=[chatbot, settings_elems["reformulate_question_cbox"], settings_elems["top_k_slider"]],
        outputs=[chatbot, last_completion],
    ).then(
        add_disclaimer,
        inputs=[last_completion, chatbot, gr.State(cfg.disclaimer)],
        outputs=[chatbot]
    ).then(
        add_sources,
        inputs=[last_completion, gr.State(max_sources)],
        outputs=[*sources_textboxes]
    ).then(
        log_completion,
        inputs=[last_completion, gr.State(cfg.MONGO_COLLECTION_INTERACTION), session_id]
    )


    flag_button.click(
        log_completion,
        inputs=[last_completion, gr.State(cfg.MONGO_COLLECTION_FLAGGED), session_id]
    ).then(
        raise_flagging_message,
    )

    # fmt: on
