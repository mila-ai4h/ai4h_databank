import copy
import json
import logging
import uuid
from datetime import datetime, timezone

import gradio as gr
import pandas as pd
from buster.busterbot import Buster
from fastapi.encoders import jsonable_encoder

import cfg
from db_utils import Feedback, init_db

mongo_db = init_db()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MAX_TABS = cfg.buster_cfg.retriever_cfg["top_k"]


def get_utc_time() -> str:
    return str(datetime.now(timezone.utc))


def get_session_id() -> str:
    return str(uuid.uuid1())


def check_auth(username: str, password: str) -> bool:
    """Basic auth, only supports a single user."""
    # TODO: update to better auth
    is_auth = username == cfg.USERNAME and password == cfg.PASSWORD
    logger.info(f"Log-in attempted. {is_auth=}")
    return is_auth


def format_sources(matched_documents: pd.DataFrame) -> list[str]:
    formatted_sources = []

    for _, doc in matched_documents.iterrows():
        formatted_sources.append(f"### [{doc.title}]({doc.url})\n{doc.content}\n")

    return formatted_sources


def pad_sources(sources: list[str]) -> list[str]:
    """Pad sources with empty strings to ensure that the number of tabs is always MAX_TABS."""
    k = len(sources)
    return sources + [""] * (MAX_TABS - k)


def chat(question, history, document_source, model, user_responses):
    history = history or []

    cfg.buster_cfg.document_source = document_source
    cfg.buster_cfg.completion_cfg["completion_kwargs"]["model"] = model
    buster.update_cfg(cfg.buster_cfg)

    response = buster.process_input(question)

    answer = response.completion.text
    history.append((question, answer))

    sources = format_sources(response.matched_documents)
    sources = pad_sources(sources)

    user_responses.append(response)

    return history, history, user_responses, *sources


def user_responses_formatted(user_responses):
    """Format user responses so that the matched_documents are in easily jsonable dict format."""
    responses_copy = copy.deepcopy(user_responses)
    for response in responses_copy:
        # go to json and back to dict so that all int entries are now strings in a dict...
        response.matched_documents = json.loads(
            response.matched_documents.drop(columns=["embedding"]).to_json(orient="index")
        )

    logger.info(responses_copy)

    return responses_copy


def submit_feedback(
    user_responses,
    session_id,
    feedback_good_bad,
    feedback_relevant_length,
    feedback_relevant_answer,
    feedback_relevant_sources,
    feedback_length_sources,
    feedback_info,
):
    dict_responses = user_responses_formatted(user_responses)
    user_feedback = Feedback(
        good_bad=feedback_good_bad,
        extra_info=feedback_info,
        relevant_answer=feedback_relevant_answer,
        relevant_length=feedback_relevant_length,
        relevant_sources=feedback_relevant_sources,
        length_sources=feedback_length_sources,
    )
    feedback = {
        "session_id": session_id,
        "user_responses": dict_responses,
        "feedback": user_feedback,
        "time": get_utc_time(),
    }
    feedback_json = jsonable_encoder(feedback)

    logger.info(feedback_json)
    try:
        mongo_db["feedback"].insert_one(feedback_json)
        logger.info("response logged to mondogb")
    except Exception as err:
        logger.exception("Something went wrong logging to mongodb")
    # update visibility for extra form
    return {feedback_submitted_message: gr.update(visible=True)}


block = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")

with block:
    buster: Buster = Buster(cfg=cfg.buster_cfg, retriever=cfg.retriever)

    # state variables are client-side and are reset every time a client refreshes the page
    user_responses = gr.State([])
    session_id = gr.State(get_session_id())

    with gr.Row():
        gr.Markdown("<h1><center>LLawMa 🦙: A Question-Answering Bot for your documentation</center></h1>")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Chatbot")
            chatbot = gr.Chatbot()
            message = gr.Textbox(
                label="Chat with 🦙",
                placeholder="Ask your question here...",
                lines=1,
            )
            submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

            with gr.Column(variant="panel"):
                gr.Markdown("## Example questions")
                with gr.Tab("Relevant questions"):
                    gr.Examples(
                        examples=[
                            "What are the key principles outlined in the OECD AI Principles, and how can they be applied in my country's legal and policy framework?",
                            "How can we ensure that AI systems are transparent, explainable, and accountable within the context of existing laws and regulations?",
                            "What are the best practices for mitigating potential biases and discrimination in AI systems to ensure fairness and inclusivity?",
                            "How can data privacy and protection be maintained while fostering AI innovation and development?",
                            "What are the key ethical considerations that should be taken into account when developing and deploying AI technologies?",
                            "How can we ensure that AI systems are used responsibly by both public and private sectors to avoid misuse or unintended consequences?",
                            "What are some examples of successful AI policy implementations in other countries or regions, and what lessons can we learn from them?",
                            "How can we encourage international cooperation and collaboration on AI policy development and standardization?",
                            "What are the potential economic, social, and environmental impacts of AI, and how can policymakers address these concerns proactively?",
                            "How can we foster a more diverse and inclusive AI workforce, and what role can policymakers play in supporting this goal?",
                            "What are some potential legal and regulatory challenges in the deployment of AI in specific sectors (e.g., healthcare, finance, transportation), and how can they be addressed?",
                            "How can we monitor and evaluate the effectiveness of AI policies and regulations over time to ensure they remain relevant and adaptable?",
                            "What role can public-private partnerships play in fostering responsible AI development and deployment, and how can we create an enabling environment for such collaborations?",
                            "How can we balance the need for AI innovation with potential concerns about job displacement or workforce transformation?",
                            "How can policymakers promote AI literacy and digital skills among the general population to ensure that everyone can benefit from AI technologies?",
                            "What are the key considerations for developing an AI policy roadmap or national AI strategy, and what are some best practices from other countries or regions?",
                            "How can we address the potential security risks and threats associated with AI technologies, such as deepfakes or autonomous weapon systems?",
                            "What are the implications of AI for intellectual property rights and how can we adapt existing laws and regulations to address these challenges?",
                            "How can policymakers encourage the development and use of AI for social good, such as addressing climate change or improving healthcare outcomes?",
                            "What are the potential implications of AI for international trade, and how can we ensure that AI does not exacerbate existing inequalities between countries?",
                            "How can we ensure that AI systems are developed and deployed in a manner that respects human rights and fundamental freedoms, including privacy and freedom of expression?",
                            "What are some approaches for establishing a regulatory or oversight framework for AI technologies that can adapt to rapid technological advancements?",
                            "How can we promote the development of AI technologies that are environmentally sustainable and energy-efficient?",
                        ],
                        inputs=message,
                        label="Questions users could ask.",
                    )
                with gr.Tab("Irrelevant questions"):
                    gr.Examples(
                        examples=[
                            "What are the most effective ways to promote a healthy work-life balance in modern society?",
                            "How can urban planning and design contribute to the creation of more sustainable and livable cities?",
                            "What role do bees play in the ecosystem, and what can be done to protect their populations from decline?",
                            "What are the most promising emerging technologies for addressing the global water crisis, and how can they be scaled up?",
                            "How has the introduction of cryptocurrencies and blockchain technology influenced the global financial landscape?",
                            "What are the key factors contributing to the rise in global obesity rates, and how can policymakers address this public health issue?",
                            "How can we promote sustainable tourism practices that benefit both local communities and the environment?",
                            "What are some effective strategies for reducing food waste at the individual, household, and institutional levels?",
                            "How has the evolution of social media impacted political discourse and the spread of information, both accurate and inaccurate?",
                            "What are some of the most significant challenges and opportunities associated with space exploration in the 21st century?",
                            "How can we encourage the use of renewable energy sources and reduce dependence on fossil fuels?",
                            "What are the key factors that contribute to the development and preservation of cultural heritage in a rapidly globalizing world?",
                            "How can we promote greater gender equality and empower women in various aspects of society, including education, employment, and political representation?",
                            "What are some innovative methods for addressing the growing issue of plastic pollution in oceans and waterways?",
                            "How can the global community work together to prevent the spread of infectious diseases and prepare for potential pandemics?",
                            "What are the key factors that influence the taste and quality of different types of wine, and how can this knowledge be used to make better wine selections?",
                            "How have advances in virtual reality and augmented reality technologies impacted the gaming industry and the way we interact with digital content?",
                            "What role do pets play in human well-being and mental health, and what are the benefits and challenges of pet ownership?",
                            "What are the primary causes of traffic congestion in urban areas, and what solutions can be implemented to alleviate this problem?",
                            "How can the arts, such as music, theater, and visual arts, contribute to the overall well-being and personal development of individuals and communities?",
                            "What are the most effective methods for learning a new language and achieving fluency, and how do these methods vary among different age groups?",
                            "What are some strategies for maintaining personal motivation and productivity while working remotely or in a flexible work environment?",
                            "How has the increasing popularity of plant-based diets influenced the food industry, and what are the potential health and environmental benefits of adopting such a diet?",
                            "What are the primary factors that contribute to the formation of natural disasters, such as earthquakes, hurricanes, and volcanic eruptions, and how can communities better prepare for these events?",
                            "How can urban agriculture and community gardens contribute to local food security and overall community well-being?",
                            "What are some innovative approaches to waste management and recycling that can help reduce the environmental impact of human activities?",
                            "How can we foster greater understanding and empathy between individuals of diverse cultural backgrounds to promote social cohesion and reduce prejudice?",
                            "What are the most effective ways to teach children about the importance of financial literacy and responsible money management?",
                            "How has the growth of e-commerce and online shopping impacted traditional brick-and-mortar retail businesses, and what strategies can they employ to stay competitive?",
                            "What are some of the most interesting and unusual hobbies or pastimes that people engage in around the world, and what can we learn from them?",
                        ],
                        inputs=message,
                        label="Questions with no relevance to the OECD AI Policy Observatory.",
                    )
                with gr.Tab("Trick questions"):
                    gr.Examples(
                        examples=[
                            "How does the United States' AI in Government Act ensure the responsible and ethical use of AI technologies in public services?",
                            "To what extent does China's Personal Information Protection Law align with international data protection standards, such as the EU's General Data Protection Regulation (GDPR)?",
                            "How does the United Kingdom's AI Sector Deal support the development of small and medium-sized AI enterprises?",
                            "In Germany's AI Liability Act, how is liability determined when multiple AI systems or a combination of AI and human decisions lead to harm or incorrect decisions?",
                            "How does France's AI Transparency and Accountability Act ensure that AI developers comply with the requirement to provide explanations for their algorithms' decision-making processes?",
                            "How does Canada's Algorithmic Impact Assessment Framework balance the benefits of AI systems in the public sector with the potential risks to individuals' rights and freedoms?",
                            "To what extent do Japan's Guidelines for AI and Data Utilization provide clear and actionable recommendations for AI developers and users?",
                            "How does Australia's Consumer Data Right (CDR) ensure that consumers can effectively exercise control over their personal data when it is used in AI applications?",
                            "What measures are included in China's New Generation Artificial Intelligence Development Plan to encourage international collaboration and knowledge-sharing in the field of AI?"
                            "In what ways does France's High Council for AI (CPAI) contribute to the global discourse on AI governance and ethics?",
                            "How does Nigeria's National AI Strategy for Nigeria (NAISN) plan to attract foreign investment and expertise to boost the country's AI capabilities?",
                            "What measures does the AI Empowerment and Inclusion Act in Nigeria propose to ensure that underserved communities have equal access to AI resources and opportunities?",
                            "How does South Africa's AI for Social Good Initiative incentivize the development of AI solutions to address the country's most pressing socio-economic challenges?",
                            "To what extent does the Data Privacy and AI Regulation Act in South Africa align with international data protection standards, such as the EU's General Data Protection Regulation (GDPR)?",
                            "How does Kenya's AI Vision 2030 plan to promote collaboration between the public and private sectors to drive AI innovation and adoption?",
                            "What specific provisions are included in Kenya's AI for Sustainable Development Act to encourage the use of AI in environmental conservation and renewable energy sectors?",
                            "How does Egypt's AI National Strategy aim to build a strong AI research and development infrastructure while fostering international partnerships?",
                            "In what ways does the AI for Smart Cities Act in Egypt support the development and implementation of AI technologies in urban planning to improve public services and infrastructure?",
                            "How does Ghana's AI Development Initiative (GAIDI) plan to leverage partnerships with regional and international AI organizations to strengthen the country's AI ecosystem?",
                            "What strategies are proposed in Ghana's AI for Healthcare Act to ensure the successful development and deployment of AI applications in healthcare services across the country?",
                            "How does Rwanda's AI Roadmap 2030 envision fostering a skilled workforce capable of driving AI innovation and adoption within the country?",
                            "How does Sweden's AI for Education and Skills Development Act plan to integrate AI technologies into educational curricula and promote training opportunities in STEM fields?",
                            "How does Brazil's National AI Strategy (BNAIS) plan to create a robust AI ecosystem through public-private partnerships and international collaborations?",
                            "What specific measures does Brazil's AI for Economic Growth and Social Inclusion Act propose to ensure that AI technologies benefit all segments of society?",
                            "How does Argentina's AI for Sustainable Agriculture Initiative support the adoption of AI technologies to enhance agricultural productivity and promote sustainable farming practices?",
                            "To what extent does the Data Privacy and AI Regulation Act in Argentina align with international data protection standards, such as the EU's General Data Protection Regulation (GDPR)?",
                            "How does Chile's AI Vision 2030 plan to promote collaboration between various stakeholders, including the government, industry, and academia, to drive AI innovation and adoption?",
                            "What specific provisions are included in Chile's AI for Renewable Energy and Environmental Conservation Act to encourage the use of AI in clean energy and environmental protection efforts?",
                            "How does Colombia's AI National Strategy aim to build a strong AI research and development infrastructure while fostering international partnerships?",
                            "In what ways does the AI for Public Health and Safety Act in Colombia support the development and implementation of AI technologies in healthcare, disaster response, and crime prevention?",
                            "How does Peru's AI Development Initiative (PADI) plan to leverage partnerships with regional and international AI organizations to strengthen the country's AI ecosystem?",
                            "What strategies are proposed in Peru's AI for Cultural Heritage Preservation Act to ensure the successful integration of AI applications in cultural preservation and documentation efforts?",
                            "How does Uruguay's AI Roadmap 2030 envision fostering a skilled workforce capable of driving AI innovation and adoption within the country?",
                            "How does India's AI for Education and Workforce Development Act plan to integrate AI technologies into educational curricula and promote training opportunities in technology and innovation fields?",
                        ],
                        inputs=message,
                        label="Questions about non-existing AI policies and laws.",
                    )

            # Feedback
            with gr.Column(variant="panel"):
                gr.Markdown("## Feedback form\nHelp us improve LLawMa 🦙!")
                with gr.Row():
                    feedback_good_bad = gr.Radio(choices=["👍", "👎"], label="How did buster do?")

                with gr.Row():
                    feedback_relevant_answer = gr.Radio(
                        choices=[
                            "1 - I lost time because the answer was wrong.",
                            "2 - I lost time because the answer was unclear.",
                            "3 - No time was saved or lost (over searching by other means).",
                            "4 - I saved time because the answer was clear and correct.",
                            "5 - The answer was perfect and can be used as a reference.",
                        ],
                        label="How much time did you save?",
                    )
                    feedback_relevant_length = gr.Radio(
                        choices=["Too Long", "Just Right", "Too Short"], label="How was the answer length?"
                    )

                with gr.Row():
                    feedback_relevant_sources = gr.Radio(
                        choices=[
                            "1 - The sources were irrelevant.",
                            "2 - The sources were relevant but others could have been better.",
                            "3 - The sources were relevant and the best ones available.",
                        ],
                        label="How relevant were the sources?",
                    )

                    feedback_length_sources = gr.Radio(
                        choices=["Too few", "Just right", "Too many"], label="How was the amount of sources?"
                    )

                feedback_info = gr.Textbox(
                    label="Enter additional information (optional)",
                    lines=10,
                    placeholder="Enter more helpful information for us here...",
                )

                submit_feedback_btn = gr.Button("Submit Feedback!")
                with gr.Column(visible=False) as feedback_submitted_message:
                    gr.Markdown("Feedback recorded, thank you! 📝")

            submit_feedback_btn.click(
                submit_feedback,
                inputs=[
                    user_responses,
                    session_id,
                    feedback_good_bad,
                    feedback_relevant_length,
                    feedback_relevant_answer,
                    feedback_relevant_sources,
                    feedback_length_sources,
                    feedback_info,
                ],
                outputs=feedback_submitted_message,
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Model")
                # TODO: remove interactive=False flag when deployed model gets access to GPT4
                model = gr.Radio(
                    cfg.available_models, label="Model to use", value=cfg.available_models[0], interactive=False
                )
            with gr.Column(scale=1):
                gr.Markdown("## Sources")
                case_names = sorted(cfg.document_sources)
                source_dropdown = gr.Dropdown(
                    choices=case_names,
                    value=case_names[0],
                    interactive=True,
                    multiselect=False,
                    label="Source",
                    info="Select a source to query",
                )
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("## References used")
                sources_textboxes = []
                for i in range(MAX_TABS):
                    with gr.Tab(f"Source {i + 1} 📝"):
                        t = gr.Markdown()
                    sources_textboxes.append(t)

    gr.Markdown("This application uses GPT to search the docs for relevant info and answer questions.")

    gr.HTML("<center> Powered by <a href='https://github.com/jerpint/buster'>Buster</a> 🤖</center>")

    state = gr.State()
    agent_state = gr.State()

    submit.click(
        chat,
        inputs=[message, state, source_dropdown, model, user_responses],
        outputs=[chatbot, state, user_responses, *sources_textboxes],
    )
    message.submit(
        chat,
        inputs=[message, state, source_dropdown, model, user_responses],
        outputs=[chatbot, state, user_responses, *sources_textboxes],
    )


block.launch(debug=True, share=False, auth=check_auth)
