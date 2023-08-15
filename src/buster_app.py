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

# Load the sample questions and split them by type
questions_file = str(current_dir / "sample_questions.csv")
questions = pd.read_csv(questions_file)
relevant_questions = questions[questions.question_type == "relevant"].question.to_list()
irrelevant_questions = questions[questions.question_type == "irrelevant"].question.to_list()
trick_questions = questions[questions.question_type == "trick"].question.to_list()


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
    with gr.Row():
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
        with gr.Column(scale=2):
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
    with gr.Column(variant="panel"):
        gr.Markdown("## Feedback form\nHelp us improve LLawMa 🦙!")
        gr.Markdown("We would love to hear from you, use this feedback form to let us know how we did.")
        with gr.Row():
            feedback_relevant_sources = gr.Radio(
                choices=["👍", "👎"], label="Were any of the retrieved sources relevant?"
            )
        with gr.Row():
            feedback_relevant_answer = gr.Radio(choices=["👍", "👎"], label="Was the generated answer useful?")

        feedback_info = gr.Textbox(
            label="Enter additional information (optional)",
            lines=10,
            placeholder="Enter more helpful information for us here...",
        )

        submit_feedback_btn = gr.Button("Submit Feedback!")
        with gr.Column(visible=False) as feedback_submitted_message:
            gr.Markdown("Feedback recorded, thank you! 📝")

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

    gr.Markdown(
        """

    ## 📚 Sources
    | Source | Report | Year | Link |
    | ---    | ---    | --- | --- |
    Laura v2|ARM/ARMENIA Unlocking Armenia's AI potential (002)/0/0|2021|/
    Laura v2|AUS/AI TECHNOLOGY ROADMAP/1/0|2019|https://data61.csiro.au/~/media/D61/AI-Roadmap-assets/19-00346_DATA61_REPORT_AI-Roadmap_WEB_191111.pdf?la=en&hash=58386288921D9C21EC8C4861CDFD863F1FBCD457
    Laura v2|AUS/AUS 2030-prosperity-through-innovation-full-report/2/0|2017|https://www.industry.gov.au/sites/default/files/May%202018/document/pdf/australia-2030-prosperity-through-innovation-full-report.pdf
    Laura v2|AUS/AUS ACOLA AI Report/3/0|2019|https://acola.org/wp-content/uploads/2019/07/hs4_artificial-intelligence-report.pdf
    Laura v2|AUS/AUS AI Standards Roadmap 20200212/5/0|2020|https://www.standards.org.au/getmedia/ede81912-55a2-4d8e-849f-9844993c3b9d/O_1515-An-Artificial-Intelligence-Standards-Roadmap-soft_1.pdf.aspx
    Laura v2|AUS/AUS AI ethics framework discussion paper/4/0|2019|https://www.csiro.au/-/media/D61/Reports/Artificial-Intelligence-ethics-framework.pdf
    Laura v2|AUS/AUS Whitepaper 2019 standards Artificial-Intelligence-Discussion-Paper-(004)/7/0|2019|https://www.standards.org.au/getmedia/aeaa5d9e-8911-4536-8c36-76733a3950d1/Artificial-Intelligence-Discussion-Paper-(004).pdf.aspx
    Laura v2|AUS/AUS tech-future/6/0|2018|Missing
    Laura v2|AUS/Australia ai-action-plan/8/0|2021|https://wp.oecd.ai/app/uploads/2021/12/Australia_AI_Action_Plan_2021.pdf
    Laura v2|AUS/INDUSTRY 4.0 TESTLAB FOR AUSTRALIA PILOT PROGRAM/10/0|2017|https://www.industry.gov.au/sites/default/files/July%202018/document/pdf/industry-4.0-testlabs-report.pdf
    Laura v2|BEL/Belgium - AI 4 BELGIUM strategy/11/0|2019|https://ai4belgium.be/wp-content/uploads/2019/04/report_en.pdf
    Laura v2|BGR/BULGARIA 2020 AI strategy white paper/12/0|2020|https://www.mtitc.government.bg/sites/default/files/conceptforthedevelopmentofaiinbulgariauntil2030.pdf
    Laura v2|BRA/BRAZIL 2021 - National AI strategy/13/0|2021|https://www.gov.br/mcti/pt-br/acompanhe-o-mcti/transformacaodigital/arquivosinteligenciaartificial/ebia-portaria_mcti_4-979_2021_anexo1.pdf
    Laura v2|CAN/CANADA 2020 AICan-2020-CIFAR-Pan-Canadian-AI-Strategy-Impact-Report/14/0|2020|https://cifar.ca/wp-content/uploads/2020/11/AICan-2020-CIFAR-Pan-Canadian-AI-Strategy-Impact-Report.pdf
    Laura v2|CAN/CANADA 2020 Cifar annualreport20182019/15/0|2019|https://cifar.ca/wp-content/uploads/2020/04/annualreport20182019.pdf
    Laura v2|CAN/CANADA 2020 Pan-Canadian-AI-Strategy-Impact-Assessment-Report 2020 Accenture/16/0|2020|https://cifar.ca/wp-content/uploads/2020/11/Pan-Canadian-AI-Strategy-Impact-Assessment-Report.pdf
    Laura v2|CAN/CANADA ai_annualreport2019_web/17/0|2019|https://cifar.ca/wp-content/uploads/2020/04/ai_annualreport2019_web.pdf
    Laura v2|CAN/CANADA rebooting-regulation-exploring-the-future-of-ai-policy-in-canada/18/0|2020|https://cifar.ca/wp-content/uploads/2020/01/rebooting-regulation-exploring-the-future-of-ai-policy-in-canada.pdf
    Laura v2|CHN/CHINA 2018 Article-Deciphering_Chinas_AI-Dream/19/0|2018|https://www.fhi.ox.ac.uk/wp-content/uploads/Deciphering_Chinas_AI-Dream.pdf
    Laura v2|CHN/CHINA AI Development Report_2018 (executive summary)/20/0|2018|https://indianstrategicknowledgeonline.com/web/China_AI_development_report_2018.pdf
    Laura v2|CHN/CHINA New Generation of Artificial Intelligence Development Plan/21/0|2017|https://flia.org/wp-content/uploads/2017/07/A-New-Generation-of-Artificial-Intelligence-Development-Plan-1.pdf
    Laura v2|CHN/CHINA_AI_standardization_white_paper_EN/22/0|2018|https://cset.georgetown.edu/wp-content/uploads/t0120_AI_standardization_white_paper_EN.pdf
    Laura v2|DEU/GERMANY 2021 Update to NAIS - Fortschreibung_KI-Strategie_engl/24/0|2020|https://www.ki-strategie-deutschland.de/files/downloads/Fortschreibung_KI-Strategie_engl.pdf
    Laura v2|DEU/GERMANY Better together - franco german cooperation on AI/25/0|2018|https://www.hertie-school.org/fileadmin/user_upload/20181218_Dt-frz-KI_Dittrich_neu.pdf
    Laura v2|DEU/GERMANY Outline_for_a_German_Artificial_Intelligence_Strategy/26/0|2018|https://www.stiftung-nv.de/sites/default/files/outline_for_a_german_artificial_intelligence_strategy.pdf
    Laura v2|DEU/Germany 20180718_Key-points_AI-Strategy_EN/23/0|2018|https://www.bmwk.de/Redaktion/EN/Downloads/E/key-points-for-federal-government-strategy-on-artificial-intelligence.pdf?__blob=publicationFile&v=1
    Laura v2|DEU/Germany Report Automated and connected driving (ethics commission) July 2019/27/0|2017|https://bmdv.bund.de/SharedDocs/EN/publications/report-ethics-commission-automated-and-connected-driving.pdf?__blob=publicationFile
    Laura v2|DNK/DENMARK Danish-Digital-Growth-Strategy2018/29/0|2018|https://eng.em.dk/media/10566/digital-growth-strategy-report_uk_web-2.pdf
    Laura v2|DNK/DENMARK Final Report from the Danish Expert Group on Data Ethics/30/0|2018|https://em.dk/media/12190/dataethics-v2.pdf
    Laura v2|DNK/DENMARK National Strategy for Artificial Intelligence 2019/31/0|2019|https://eng.em.dk/media/13081/305755-gb-version_4k.pdf
    Laura v2|DNK/Denmark 2018 Recommendations_when_using_supervised_ML pdf/28/0|2018|https://www.dfsa.dk/-/media/Tilsyn/Recommendations_when_using_supervised_ML-pdf.pdf
    Laura v2|ESP/National AI Strategy/32/0|2020|https://www.lamoncloa.gob.es/presidente/actividades/Documents/2020/ENIA2B.pdf
    Laura v2|EST/AI Report 2019/33/0|2019|https://www.ria.ee/en/media/580/download
    Laura v2|EU/AI, DATA AND ROBOTICS PARTNERSHIP IN HORIZON EUROPE/34/0|2020|https://bdva.eu/sites/default/files/AI-Data-Robotics-Partnership-SRIDA%20V3.1.pdf
    Laura v2|EU/COORDINATED PLAN ON ARTIFICIAL INTELLIGENCE/35/0|2018|https://ec.europa.eu/commission/presscorner/api/files/document/print/en/ip_18_6689/IP_18_6689_EN.pdf
    Laura v2|EU/EU 2020 AI Investment - FINAL/36/0|2020|https://core.ac.uk/download/pdf/322748045.pdf
    Laura v2|EU/EU 2020 Shaping_Europe_s_digital_future___Questions_and_Answers/37/0|2020|https://ec.europa.eu/commission/presscorner/api/files/document/print/en/qanda_20_264/QANDA_20_264_EN.pdf
    Laura v2|EU/EU 2021 communication-digital-compass-2030_en/38/0|2021|https://eur-lex.europa.eu/resource.html?uri=cellar:12e835e2-81af-11eb-9ac9-01aa75ed71a1.0001.02/DOC_1&format=PDF
    Laura v2|EU/EU EGE statement on ethics 2018/39/0|2018|https://op.europa.eu/en/publication-detail/-/publication/6b1bc507-af70-11e8-99ee-01aa75ed71a1/language-en/format-PDF
    Laura v2|EU/EU EPSC Strategic Note on AI/40/0|2018|https://op.europa.eu/en/publication-detail/-/publication/f22f6811-1007-11ea-8c1f-01aa75ed71a1/language-en
    Laura v2|EU/EU Network of AI Excellence Centres/41/0|2020|https://umai.uma.es/shared/Commission-Presentation-H2020.pdf
    Laura v2|EU/EU Parlamient 2020 - Opportunities of AI report/42/0|2020|https://www.europarl.europa.eu/RegData/etudes/STUD/2020/652713/IPOL_STU(2020)652713_EN.pdf
    Laura v2|EU/EU Report Liability for AI and other digital technologies/43/0|2019|https://www.europarl.europa.eu/meetdocs/2014_2019/plmrep/COMMITTEES/JURI/DV/2020/01-09/AI-report_EN.pdf
    Laura v2|EU/OPEN DATA DIRECTIVE/44/0|2019|https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32019L1024
    Laura v2|EU/REPORT ON SAFETY AND LIABILITY IMPLICATIONS OF ARTIFICIAL INTELLIGENCE, THE INTERNET OF THINGS AND ROBOTICS/45/0|2020|https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:52020DC0064
    Laura v2|EU/THE ROBUSTNESS AND EXPLAINABILITY OF ARTIFICIAL INTELLIGENCE/46/0|2020|https://publications.jrc.ec.europa.eu/repository/bitstream/JRC119336/dpad_report.pdf
    Laura v2|EU/WHITE PAPER ON ARTIFICIAL INTELLIGENCE/47/0|2020|https://commission.europa.eu/system/files/2020-02/commission-white-paper-artificial-intelligence-feb2020_en.pdf
    Laura v2|FIN/Finland - Age of Artificial Intellilgence/48/0|2017|https://julkaisut.valtioneuvosto.fi/bitstream/handle/10024/160391/TEMrap_47_2017_verkkojulkaisu.pdf
    Laura v2|FIN/Finland - Work in the age of artificial intelligence/49/0|2018|https://julkaisut.valtioneuvosto.fi/bitstream/handle/10024/160980/TEMjul_21_2018_Work_in_the_age.pdf
    Laura v2|FRA/For a meaningful artifical intelligence- towards a french and european strategy/50/0|2018|https://www.aiforhumanity.fr/pdfs/MissionVillani_Report_ENG-VF.pdf
    Laura v2|GBR/National AI strategy/51/0|2021|https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1020402/National_AI_Strategy_-_PDF_version.pdf
    Laura v2|GBR/The Pathway to Driverless Cars/52/0|2015|https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/401562/pathway-driverless-cars-summary.pdf
    Laura v2|GBR/UK - AI in the UK - ready willing and able/53/0|2018|https://publications.parliament.uk/pa/ld201719/ldselect/ldai/100/100.pdf
    Laura v2|GBR/UK DfE-Education_Technology_Strategy/54/0|2019|https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/791931/DfE-Education_Technology_Strategy.pdf
    Laura v2|GBR/UK Growing_the_artificial_intelligence_industry_in_the_UK/55/0|2019|https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/652097/Growing_the_artificial_intelligence_industry_in_the_UK.pdf
    Laura v2|GBR/UK Initial code of conduct for data-driven health and care technology/56/0|2018|https://www.gov.uk/government/publications/code-of-conduct-for-data-driven-health-and-care-technology/initial-code-of-conduct-for-data-driven-health-and-care-technology
    Laura v2|GBR/UK SECTOR DEAL (AI) 180425_BEIS/58/0|2018|https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/702810/180425_BEIS_AI_Sector_Deal__4_.pdf
    Laura v2|GBR/UK report 2020 House of Parliaments AI in the UK - No room for complacency/57/0|2020|https://publications.parliament.uk/pa/ld5801/ldselect/ldliaison/196/196.pdf
    Laura v2|IND/INDIA NationalStrategy-for-AI-Discussion-Paper/59/0|2018|https://indiaai.gov.in/documents/pdf/NationalStrategy-for-AI-Discussion-Paper.pdf
    Laura v2|ITA/ITALY 2019 Strategia-Nazionale-Intelligenza-Artificiale-Bozza-Consultazione/60/0|2019|https://www.mimit.gov.it/images/stories/documenti/Strategia-Nazionale-Intelligenza-Artificiale-Bozza-Consultazione.pdf
    Laura v2|ITA/ITALY whitepaper/61/0|2018|https://ia.italia.it/assets/whitepaper.pdf
    Laura v2|ITA/Proposte per una Strategia italiana per l'intelligenza artificiale/62/0|2020|https://www.mimit.gov.it/images/stories/documenti/Proposte_per_una_Strategia_italiana_AI.pdf
    Laura v2|JPN/JAPAN 2021 Governance Guidelines for Implementation of AI Principles/63/0|2021|file:///D:/Users/jans/Desktop/data/JPN/JAPAN%202021%20Governance%20Guidelines%20for%20Implementation%20of%20AI%20Principles/JAPAN%202021%20Governance%20Guidelines%20for%20Implementation%20of%20AI%20Principles.pdf
    Laura v2|JPN/JAPAN AI Utilization principles explained/64/0|2018|https://www.soumu.go.jp/main_content/000581310.pdf
    Laura v2|JPN/JAPAN Integrated Innovation Strategy 2018/65/0|2018|https://www8.cao.go.jp/cstp/english/doc/integrated_main.pdf
    Laura v2|JPN/JAPAN MIC AI R&D Guidelines/66/0|2017|https://www.soumu.go.jp/main_content/000507517.pdf
    Laura v2|JPN/JAPAN Social Principles of Human-centric AI/67/0|2019|https://www8.cao.go.jp/cstp/stmain/aisocialprinciples.pdf
    Laura v2|LUX/Luxembourg AI strategic vision for Luxembourg/68/0|2019|https://gouvernement.lu/dam-assets/fr/publications/rapport-etude-analyse/minist-digitalisation/Artificial-Intelligence-a-strategic-vision-for-Luxembourg.pdf
    Laura v2|LUX/Luxembourg The-Data-driven-Innovation-Strategy/69/0|2019|https://gouvernement.lu/dam-assets/fr/publications/rapport-etude-analyse/minist-economie/The-Data-driven-Innovation-Strategy.pdf
    Laura v2|MEX/MEXICO - Towards an AI strategy in Mexico/70/0|2018|https://go.wizeline.com/rs/571-SRN-279/images/Towards-an-AI-strategy-in-Mexico.pdf
    Laura v2|MEX/MEXICO Consolidado_Comentarios_Consulta_IA__1_/71/0|2019|https://www.gob.mx/cms/uploads/attachment/file/415644/Consolidado_Comentarios_Consulta_IA__1_.pdf
    Laura v2|MLT/MALTA - AI strategy/72/0|2019|https://malta.ai/wp-content/uploads/2019/04/Draft_Policy_document_-_online_version.pdf
    Laura v2|NLD/NETHERLANDS - 2019 AIREA-NL+AI+Research+Agenda+for+the+Netherlands/74/0|2019|https://www.nwo.nl/sites/nwo/files/documents/AIREA-NL%20AI%20Research%20Agenda%20for%20the%20Netherlands.pdf
    Laura v2|NLD/Netherlands 2021 min-ezk-digitaliseringstrategie-en-v03/75/0|2021|https://www.nederlanddigitaal.nl/binaries/nederlanddigitaal-nl/documenten/publicaties/2021/06/22/the-dutch-digitalisation-strategy-2021-eng/210621-min-ezk-digitaliseringstrategie-en-v03.pdf
    Laura v2|NLD/Netherlands Rapport-AI-voor-Nederland_181106_105304 (DUTCH)/77/0|2018|https://www.vno-ncw.nl/sites/default/files/aivnl_20181106_0.pdf
    Laura v2|NOR/NORWAY 2020 AI STRATEGY/78/0|2020|https://www.regjeringen.no/contentassets/1febbbb2c4fd4b7d92c67ddd353b6ae8/en-gb/pdfs/ki-strategi_en.pdf
    Laura v2|NZL/ALGORITHM ASSESSMENT REPORT/79/0|2018|https://www.data.govt.nz/assets/Uploads/Algorithm-Assessment-Report-Oct-2018.pdf
    Laura v2|NZL/DIGITAL ECONOMY PARTNERSHIP AGREEMENT (NEW ZEALAND, SINGAPORE AND CHILE)/80/1|2020|https://www.mti.gov.sg/-/media/MTI/Microsites/DEAs/Digital-Economy-Partnership-Agreement/Digital-Economy-Partnership-Agreement.pdf
    Laura v2|OECD/OECD G20 examples-of-ai-national-policies/81/0|2020|https://www.oecd.org/sti/examples-of-ai-national-policies.pdf
    Laura v2|POL/POLAND 2020 Policy for the Development of Artificial Intelligence in Poland 2020/82/0|2020|https://wp.oecd.ai/app/uploads/2021/12/Poland_Policy_for_Artificial_Intelligence_Development_in_Poland_from_2020_2020.pdf
    Laura v2|PRT/PORTUGAL 2020 APRIL - ZONAS LIBRES TECNOLOGICAS/83/0|2020|https://files.dre.pt/1s/2020/04/07800/0000600032.pdf
    Laura v2|PRT/PORTUGAL INCODE 2030 Matos InnovWSIS-V1/84/0|2018|https://www.itu.int/en/ITU-D/Regional-Presence/Europe/Documents/Events/2018/WSIS/Matos%20InnovWSIS-V1.pdf
    Laura v2|PRT/PORTUGAL incode2030_en/85/0|2021|https://incode2030.pt/sites/default/files/incode2030_en_0.pdf
    Laura v2|SAU/SAUDI KINETIC consulting Designing-A-Winning-AI-Strategy/86/0|2017|https://www.kineticcs.com/wp-content/uploads/2017/11/Designing-A-Winning-AI-Strategy-_Kinetic-Consulting-Services.pdf
    Laura v2|SAU/SAUDI PWC economic-potential-ai-middle-east/87/0|2018|https://www.pwc.com/m1/en/publications/documents/economic-potential-ai-middle-east.pdf
    Laura v2|SAU/SAUDI_Vision2030_EN/88/0|2019|https://www.saudiembassy.net/sites/default/files/u66/Saudi_Vision2030_EN.pdf
    Laura v2|SGP/Model Aritficial Intelligence Governance Framework Second edition/89/0|2020|https://www.pdpc.gov.sg/-/media/files/pdpc/pdf-files/resource-for-organisation/ai/sgmodelaigovframework2.pdf
    Laura v2|SGP/Singapore Compendium of Use Cases Vol 2/90/0|2020|https://file.go.gov.sg/ai-gov-use-cases-2.pdf
    Laura v2|SRB/SERBIA - National AI Strategy DEC 2019/91/0|2019|https://www.media.srbija.gov.rs/medsrp/dokumenti/strategy_artificial_intelligence.pdf
    Laura v2|SVK/SLOVAKIA 2019 AP-DT-English-Version-FINAL/92/0|2019|https://www.mirri.gov.sk/wp-content/uploads/2019/10/AP-DT-English-Version-FINAL.pdf
    Laura v2|SVN/NpUI-SI-2025/93/0|2021|https://www.gov.si/assets/ministrstva/MJU/DID/NpUI-SI-2025.docx
    Laura v2|UN/UN Resource Guide on AI Strategies_April 2021_rev_0/95/0|2021|https://sdgs.un.org/sites/default/files/2021-06/Resource%20Guide%20on%20AI%20Strategies_June%202021.pdf
    Laura v2|USA/AI Principles Recommendations on the Ethical Use of Artificial Intelligence by the Department of Defense/96/0|2019|https://media.defense.gov/2019/Oct/31/2002204458/-1/-1/0/DIB_AI_PRINCIPLES_PRIMARY_DOCUMENT.PDF
    Laura v2|USA/AUTOMATED VEHICLES 3.0 PREPARING FOR THE FUTURE OF TRANSPORTATION/97/0|2019|https://www.transportation.gov/sites/dot.gov/files/docs/policy-initiatives/automated-vehicles/320711/preparing-future-transportation-automated-vehicle-30.pdf
    Laura v2|USA/FDA PROPOSED REGULATORY FRAMEWORK FOR MODIFICATIONS TO AIML BASED SOFTWARE AS A MEDICAL DEVICE/98/0|2021|https://www.fda.gov/media/145022/download
    Laura v2|USA/FEDERAL DATA STRATEGY ACTION PLAN/99/0|2019|https://strategy.data.gov/assets/docs/draft-2019-2020-federal-data-strategy-action-plan.pdf
    Laura v2|USA/FTC CONSUMER PROTECTION AND COMPETITION INVESTIGATIONS (Bias in Algorithms and Biometrics)/100/0|2021|https://www.ftc.gov/system/files/attachments/press-releases/ftc-streamlines-consumer-protection-competition-investigations-eight-key-enforcement-areas-enable/omnibus_resolutions_p859900.pdf
    Laura v2|USA/NATIONAL AI R&D STRATEGIC PLAN/101/0|2019|https://www.nitrd.gov/pubs/National-AI-RD-Strategy-2019.pdf
    Laura v2|USA/NATIONAL ROBOTICS INITIATIVE 2.0 UBIQUITOUS COLLABORATIVE ROBOTS/102/0|2021|https://www.nsf.gov/pubs/2021/nsf21559/nsf21559.pdf
    Laura v2|USA/PROTECTING THE UNITED STATES ADVANTAGE IN ARTIFICIAL INTELLIGENCE AND RELATED CRITICAL TECHNOLOGIES/103/0|2018|https://dod.defense.gov/Portals/1/Documents/pubs/2018-National-Defense-Strategy-Summary.pdf
    Laura v2|USA/SUMMARY OF THE 2018 DEPARTMENT OF DEFENSE ARTIFICIAL INTELLIGENCE STRATEGY/105/0|2018|https://media.defense.gov/2019/Feb/12/2002088963/-1/-1/1/SUMMARY-OF-DOD-AI-STRATEGY.PDF
    Laura v2|USA/Strengthening-International-Cooperation-AI_Oct21/104/0|2021|https://www.brookings.edu/wp-content/uploads/2021/10/Strengthening-International-Cooperation-AI_Oct21.pdf
    Laura v2|USA/THE AIM INITIATIVE A STRATEGY FOR AUGMENTING INTELLIGENCE USING MACHINES/106/0|2019|https://www.dni.gov/files/ODNI/documents/AIM-Strategy.pdf
    Laura v2|USA/US 2019 Draft-OMB-Memo-on-Regulation-of-AI-1-7-19/107/0|2020|https://www.whitehouse.gov/wp-content/uploads/2020/01/Draft-OMB-Memo-on-Regulation-of-AI-1-7-19.pdf
    Laura v2|USA/US 2019 Standards ai_standards_fedengagement_plan_9aug2019/108/0|2019|https://www.nist.gov/system/files/documents/2019/08/10/ai_standards_fedengagement_plan_9aug2019.pdf
    Laura v2|USA/US 2020 Recommendations-Cloud-AI-RD-Nov2020/109/0|2020|https://www.nitrd.gov/pubs/Recommendations-Cloud-AI-RD-Nov2020.pdf
    Laura v2|USA/US Administration-2017-ST-Highlights/110/0|2017|https://trumpwhitehouse.archives.gov/wp-content/uploads/2018/03/Administration-2017-ST-Highlights.pdf
    Laura v2|USA/US Artificial-Intelligence-Automation-Economy/111/0|2016|https://obamawhitehouse.archives.gov/sites/whitehouse.gov/files/documents/Artificial-Intelligence-Automation-Economy.PDF
    Laura v2|USA/US Executive Order - Maintaining American Leadership in Artificial Intelligence/112/0|2019|https://www.govinfo.gov/content/pkg/FR-2019-02-14/pdf/2019-02544.pdf
    Laura v2|USA/US Feb 2020 American-AI-Initiative-One-Year-Annual-Report/113/0|2020|https://www.nitrd.gov/nitrdgroups/images/c/c1/American-AI-Initiative-One-Year-Annual-Report.pdf
    Laura v2|USA/US Sept 2019 - FY2020-NITRD-AI-RD-Budget-September-2019/114/0|2019|https://www.nitrd.gov/pubs/FY2020-NITRD-Supplement.pdf
    Laura v2|USA/US Summary-Report-of-White-House-AI-Summit/116/0|2018|https://www.nitrd.gov/nitrdgroups/images/2/23/Summary-Report-of-White-House-AI-Summit.pdf
    Laura v2|WEF/WEF_National_AI_Strategy/117/0|2019|https://www3.weforum.org/docs/WEF_National_AI_Strategy.pdf?_gl=1*1ycz2tk*_up*MQ..&gclid=EAIaIQobChMI2LuIqrL1_gIVSazVCh3WZA_0EAAYASAAEgKY9fD_BwE
    LimeSurvey v2|ARG/Laura/0/0|2021|https://publications.iadb.org/publications/english/document/Artificial-Intelligence-for-Social-Good-in-Latin-America-and-the-Caribbean-The-Regional-Landscape-and-12-Country-Snapshots.pdf
    LimeSurvey v2|ARG/Prometea/1/0|2021|https://giswatch.org/sites/default/files/gisw2019_artificial_intelligence.pdf
    LimeSurvey v2|AUS/R_1515 An Artificial Intelligence Standards Roadmap softpdf/2/0|2021|https://www.standards.org.au/getmedia/ede81912-55a2-4d8e-849f-9844993c3b9d/R_1515-An-Artificial-Intelligence-Standards-Roadmap-soft.pdf.aspx
    LimeSurvey v2|CHE/A legal framework for artificial intelligence/4/0|2022|https://www.dsi.uzh.ch/dam/jcr:3a0cb402-c3b3-4360-9332-f800895fdc58/dsi-strategy-lab-21-de.pdf
    LimeSurvey v2|CHE/Digital Switzerland Strategy/5/0|2019|https://ethicsandtechnology.org/wp-content/uploads/2019/12/bericht_idag_ki_d.pdf
    LimeSurvey v2|CHL/documento_politica_ia_digital/6/0|2019|https://minciencia.gob.cl/uploads/filer_public/bc/38/bc389daf-4514-4306-867c-760ae7686e2c/documento_politica_ia_digital_.pdf
    LimeSurvey v2|COL/Conpes 3975/7/0|2021|https://colaboracion.dnp.gov.co/CDT/Conpes/Económicos/3975.pdf
    LimeSurvey v2|COL/Emerging Technologies Handboo/8/0|2021|https://gobiernodigital.mintic.gov.co/692/articles-160829_Guia_Tecnologias_Emergentes.pdf
    LimeSurvey v2|COL/NATIONAL ENTREPRENEURSHIP POLICY CONPES 4011/9/0|2021|https://colaboracion.dnp.gov.co/CDT/Conpes/Económicos/4011.pdf
    LimeSurvey v2|EGY/Egypts National AI Strategy/12/0|2021|https://mcit.gov.eg/Upcont/Documents/Publications_672021000_Egypt-National-AI-Strategy-English.pdf
    LimeSurvey v2|ESP/Spanish RDI Strategy In Artificial Intelligence/14/0|2021|https://www.ciencia.gob.es/dam/jcr:5af98ba2-166c-4e63-9380-4f3f68db198e/Estrategia_Inteligencia_Artificial_IDI.pdf
    LimeSurvey v2|EU/Common Regulatory Capacity for AI/15/0|2022|https://www.turing.ac.uk/sites/default/files/2022-07/common_regulatory_capacity_for_ai_the_alan_turing_institute.pdf
    LimeSurvey v2|FIN/Data Protection Ombudsman Fines Kymen Vesi Oy/16/0|2022|https://tietosuoja.fi/documents/6927448/22406974/Ty%C3%B6ntekij%C3%B6iden+sijaintitietojen+k%C3%A4sittely+ja+vaikutustenarviointi.pdf/2d04e545-d427-8a0d-3f4d-967de7b428ac/Ty%C3%B6ntekij%C3%B6iden+sijaintitietojen+k%C3%A4sittely+ja+vaikutustenarviointi.pdf
    LimeSurvey v2|GBR/AI Ecosystem Survey/17/0|2021|https://www.turing.ac.uk/sites/default/files/2021-09/ai-strategy-survey_results_020921.pdf
    LimeSurvey v2|GBR/AI in Financial Services/18/0|2021|https://www.turing.ac.uk/sites/default/files/2021-06/ati_ai_in_financial_services_lores.pdf
    LimeSurvey v2|IND/Approach Document/20/0|2021|https://www.niti.gov.in/sites/default/files/2021-02/Responsible-AI-22022021.pdf
    LimeSurvey v2|IRL/National Landcover mapping project/21/0|2021|https://www.npws.ie/sites/default/files/general/NLCHM%20Newsletter%20May%202017.pdf
    LimeSurvey v2|IRL/Public Consultation for the National AI Strategy for Ireland/22/0|2021|https://enterprise.gov.ie/en/Consultations/Consultations-files/AI-Strategy-Public-Consultation-Report.pdf
    LimeSurvey v2|IRL/Satellite Platform for Ireland SPEir/23/0|2021|http://eoscience.esa.int/landtraining2018/files/posters/hanafin.pdf
    LimeSurvey v2|ISR/National Initiative for Secured Intelligent Systems/24/0|2022|https://icrc.tau.ac.il/sites/cyberstudies-english.tau.ac.il/files/media_server/cyber%20center/The%20National%20Initiative_eng%202021_digital.pdf
    LimeSurvey v2|ITA/Artificial Intelligence Strategic Programme 2022 2024/25/0|2021|https://www.mise.gov.it/images/stories/documenti/Strategia-Nazionale-Intelligenza-Artificiale-Bozza-Consultazione.pdf
    LimeSurvey v2|ITA/N RG 29492019/26/0|2022|https://www.bollettinoadapt.it/wp-content/uploads/2021/01/Ordinanza-Bologna.pdf
    LimeSurvey v2|JPN/Governing Innovation/27/0|2021|https://www.meti.go.jp/press/2020/07/20200713001/20200713001-2.pdf
    LimeSurvey v2|KEN/Kenyas Digital Economy Strategy/28/0|2021|https://ict.go.ke/wp-content/uploads/2020/08/10TH-JULY-FINAL-COPY-DIGITAL-ECONOMY-STRATEGY-DRAFT-ONE.pdf
    LimeSurvey v2|MUS/Mauritius AI Council/29/0|2022|https://ncb.govmu.org/ncb/strategicplans/MauritiusAIStrategy2018.pdf
    LimeSurvey v2|POL/Amendments to the Polish Labour Code/31/0|2022|http://ilo.org/dyn/natlex/docs/ELECTRONIC/45181/91758/F1623906595/The-Labour-Code%20consolidated%201997.pdf
    LimeSurvey v2|THA/12th National Economic and Social Development Plan/33/0|2019|https://www.oneplanetnetwork.org/sites/default/files/thailand_national_economic_and_social_development_plan_nesdp.pdf
    LimeSurvey v2|THA/Ministry of Digital Economy and Society/34/0|2020|https://www.etda.or.th/getattachment/9d370f25-f37a-4b7c-b661-48d2d730651d/Digital-Thailand-AI-Ethics-Principle-and-Guideline.pdf.aspx?lang=th-TH
    LimeSurvey v2|TUR/Action Plan/35/0|2022|https://inhak.adalet.gov.tr/Resimler/SayfaDokuman/1262021081047Action_Plan_On_Human_Rights.pdf
    LimeSurvey v2|TUR/PhD Scholarship/36/0|2022|https://www.yok.gov.tr/Documents/Yayinlar/Yayinlarimiz/2020/100-2000-yok-doktora-projesi-2020.pdf
    LimeSurvey v2|TUR/TRNationalAIStrategy2021 2025/38/0|2022|https://cbddo.gov.tr/SharedFolderServer/Genel/File/TRNationalAIStrategy2021-2025.pdf
    LimeSurvey v2|TUR/Toolkit/37/0|2022|https://www3.weforum.org/docs/WEF_Human_Centred_Artificial_Intelligence_for_Human_Resources_2021.pdf
    LimeSurvey v2|UAE/UAE National Strategy for AI 2031/39/0|2019|https://ai.gov.ae/wp-content/uploads/2021/07/UAE-National-Strategy-for-Artificial-Intelligence-2031.pdf
    LimeSurvey v2|URY/Predpol/40/0|2021|https://www.minterior.gub.uy/images/2017/Noviembre/Cmo-evitar-el-delito-urbano.pdf
    LimeSurvey v2|USA/2020 Federal Data Strategy Action Plan/41/0|2022|https://strategy.data.gov/assets/docs/2020-federal-data-strategy-action-plan.pdf
    LimeSurvey v2|USA/AI and Society/42/0|2022|https://www.nsf.gov/pubs/2019/nsf19018/nsf19018.pdf
    LimeSurvey v2|USA/FRVT Demographic Effects/46/0|2021|https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.8280.pdf
    LimeSurvey v2|USA/Fairness Ethics Accountability and Transparency/43/0|2022|https://www.nsf.gov/pubs/2019/nsf19016/nsf19016.pdf
    LimeSurvey v2|USA/Federal 5 Year STEM Education Strategic Plan/44/0|2019|https://www.energy.gov/sites/default/files/2019/05/f62/STEM-Education-Strategic-Plan-2018.pdf
    LimeSurvey v2|USA/Federal Data Strategy/45/0|2022|https://strategy.data.gov/assets/docs/2020-federal-data-strategy-framework.pdf
    LimeSurvey v2|USA/Guidance for Regulation of AI Applications/48/0|2020|https://www.whitehouse.gov/wp-content/uploads/2020/11/M-21-06.pdf
    LimeSurvey v2|USA/NITRD NAIIO Supplement to the Presidents FY2022 Budget/55/0|2021|https://www.nitrd.gov/pubs/FY2022-NITRD-NAIIO-Supplement.pdf
    LimeSurvey v2|USA/National AI Initiative Act of 2020/49/0|2021|https://www.congress.gov/116/crpt/hrpt617/CRPT-116hrpt617.pdf#page=1210
    LimeSurvey v2|USA/National AI RD Strategic Plan/50/0|2022|https://www.nitrd.gov/pubs/national_ai_rd_strategic_plan.pdf
    LimeSurvey v2|USA/National Defense Authorization Act for Fiscal Year 2021/52/0|2022|https://www.govinfo.gov/content/pkg/BILLS-116hr6395enr/pdf/BILLS-116hr6395enr.pdf
    LimeSurvey v2|USA/National Institute of Standards and Technology Principles for Explainable AI/53/0|2021|https://www.nist.gov/system/files/documents/2020/08/17/NIST%20Explainable%20AI%20Draft%20NISTIR8312%20%281%29.pdf
    LimeSurvey v2|USA/National Strategy for Critical and Emerging Technologies/54/1|2020|https://trumpwhitehouse.archives.gov/wp-content/uploads/2020/10/National-Strategy-for-CET.pdf
    LimeSurvey v2|USA/Policy on No Action Letters/58/0|2019|https://files.consumerfinance.gov/f/documents/cfpb_final-policy-on-no-action-letters.pdf
    LimeSurvey v2|USA/Policy on the Compliance Assistance Sandbox/59/0|2022|https://files.consumerfinance.gov/f/documents/cfpb_final-policy-on-cas.pdf
    LimeSurvey v2|USA/Policy to Encourage Trial Disclosure Programs/60/0|2022|https://files.consumerfinance.gov/f/documents/cfpb_final-policy-to-encourage-tdp.pdf
    LimeSurvey v2|USA/US Air Force AI Annex to the Department of Defense AI Strategy/64/0|2019|https://www.af.mil/Portals/1/documents/5/USAF-AI-Annex-to-DoD-AI-Strategy.pdf
    LimeSurvey v2|USA/White House Summit on AI for American Industry/65/0|2022|https://trumpwhitehouse.archives.gov/wp-content/uploads/2018/05/Summary-Report-of-White-House-AI-Summit.pdf
    LimeSurvey v2|USA/White House Summit on AI in Government/66/0|2019|https://trumpwhitehouse.archives.gov/wp-content/uploads/2019/09/Summary-of-White-House-Summit-on-AI-in-Government-September-2019.pdf
    Orsolya|OECD/A Blueprint for Building National Compute Capacity for Artificial Intelligence/0/0|2023|https://www.oecd-ilibrary.org/docserver/876367e3-en.pdf?expires=1684152198&id=id&accname=guest&checksum=7C7A5A59453642FE2A7474BFA0EB274A
    Orsolya|OECD/A portrait of AI adopters across countries- Firm characteristics, assets’ complementarities and productivity/1/0|2023|https://www.oecd-ilibrary.org/docserver/0fb79bb9-en.pdf?expires=1684152244&id=id&accname=guest&checksum=721A5E04352D70A597485F978931F485
    Orsolya|OECD/AI Language Models Technological, Socio-Economic and Policy Considerations/5/0|2021|https://www.oecd-ilibrary.org/docserver/13d38f92-en.pdf?expires=1684154360&id=id&accname=guest&checksum=BA3F1E3502461B6DFE8F0331A1DB2172
    Orsolya|OECD/AI Measurement in ICT usage Surveys a Review/6/0|2021|https://www.oecd-ilibrary.org/docserver/72cce754-en.pdf?expires=1684154430&id=id&accname=guest&checksum=AB65E3F8DFB44DB81357FDE38ABF3DDB
    Orsolya|OECD/AI and the Future of Skills, Volume 1/3/0|2021|https://www.oecd-ilibrary.org/docserver/5ee71f34-en.pdf?expires=1684154218&id=id&accname=guest&checksum=2D49DA705C6CC47C2A300FB596DA1854
    Orsolya|OECD/AI in Business and Finance OECD business and finance outlook 2021/4/0|2021|https://www.oecd-ilibrary.org/deliver/ba682899-en.pdf?itemId=/content/publication/ba682899-en&mimeType=pdf
    Orsolya|OECD/Advancing Accountability in AI Governing And Managing Risks throughout the Lifecycle for Trustworthy AI/2/0|2023|https://www.oecd-ilibrary.org/docserver/2448f04b-en.pdf?expires=1684154140&id=id&accname=guest&checksum=0C0F4C1AF8CF381EBB0571169E9AD359
    Orsolya|OECD/An overview of national AI strategies and policies/7/0|2021|https://www.oecd-ilibrary.org/docserver/c05140d9-en.pdf?expires=1684154480&id=id&accname=guest&checksum=F13FCA30959AA49F5D4C757CB5629EE8
    Orsolya|OECD/Artificial Intelligence in Society/9/0|2021|https://www.oecd-ilibrary.org/deliver/eedfee77-en.pdf?itemId=%2Fcontent%2Fpublication%2Feedfee77-en&mimeType=pdf
    Orsolya|OECD/Artificial intelligence and employment New cross-country evidence/8/0|2021|https://www.oecd-ilibrary.org/docserver/c2c1d276-en.pdf?expires=1684154518&id=id&accname=guest&checksum=105E0EFA8615968D9AAF4DD573AB9C00
    Orsolya|OECD/Demand for AI skills in jobs- Evidence from online job postings/10/0|2021|https://www.oecd-ilibrary.org/docserver/3ed32d94-en.pdf?expires=1684154629&id=id&accname=guest&checksum=F1A59E44AA287750E8242E28FA7DF405
    Orsolya|OECD/Examples of LAC AI instruments aligned with OECD AI values-based principles/11/0|2021|https://www.oecd-ilibrary.org/sites/5fbd4ea6-en/index.html?itemId=/content/component/5fbd4ea6-en
    Orsolya|OECD/Identifying and characterising AI adopters A novel approach based on big data/12/0|2022|https://www.oecd-ilibrary.org/docserver/154981d7-en.pdf?expires=1684154767&id=id&accname=guest&checksum=FB38E677D12C7A4E20744C254AFC1DA8
    Orsolya|OECD/Identifying artificial intelligence actors using online data/13/0|2023|https://www.oecd-ilibrary.org/docserver/1f5307e7-en.pdf?expires=1684154827&id=id&accname=guest&checksum=8ED6066AA997166A5B0EB960576A52B9
    Orsolya|OECD/Is Education Losing the Race with Technology/14/0|2023|https://www.oecd-ilibrary.org/docserver/73105f99-en.pdf?expires=1684154877&id=id&accname=guest&checksum=7A51F751B05491967D9FB32DE9F31DE5
    Orsolya|OECD/Measuring the AI content of government-funded R&D projects A proof of concept for the OECD Fundstat initiative/15/0|2021|https://www.oecd-ilibrary.org/docserver/7b43b038-en.pdf?expires=1684154919&id=id&accname=guest&checksum=9E8322AC72291582CEA0487FCFA8D997
    Orsolya|OECD/Measuring the Environmental Impacts of Artificial Intelligence Compute And Applications- The Ai Footprint/16/0|2022|https://www.oecd-ilibrary.org/docserver/7babf571-en.pdf?expires=1684155261&id=id&accname=guest&checksum=D48E0A8AB5D878A431342E7D48C5D505
    Orsolya|OECD/OECD Framework for the Classification of AI Systems/17/0|2022|https://www.oecd-ilibrary.org/docserver/cb6d9eca-en.pdf?expires=1684155498&id=id&accname=guest&checksum=1DDDDDD67BFEE97D0B2E663552C3701D
    Orsolya|OECD/Opportunities and drawbacks of using artificial intelligence for training/18/0|2021|https://www.oecd-ilibrary.org/docserver/22729bd6-en.pdf?expires=1684155337&id=id&accname=guest&checksum=91C6B629B8AD7D2BD53E09008A256DD4
    Orsolya|OECD/Review of National Policy Initiatives in Support of Digital and AI-Driven Innovation/19/0|2019|https://www.oecd-ilibrary.org/docserver/15491174-en.pdf?expires=1684155400&id=id&accname=guest&checksum=F704E2B7E5B702199036BBC50963FC96
    Orsolya|OECD/Scoping the OECD AI Principles Deliberations of the Expert Group On Artificial Intelligence at the OECD (AIGO)/20/0|2019|https://www.oecd-ilibrary.org/docserver/d62f618a-en.pdf?expires=1684155536&id=id&accname=guest&checksum=131DB451146273AFD8D45AF3D49B73DC
    Orsolya|OECD/State Of Implementation Of The OECD AI Principles Insights From National AI Policies/21/0|2021|https://www.oecd-ilibrary.org/docserver/1cd40c44-en.pdf?expires=1684154982&id=id&accname=guest&checksum=4C48996826131FE53EDBE1E08974A5CB
    Orsolya|OECD/The Human Capital Behind AI- Jobs and Skills Demand from Online Job Postings/22/0|2021|https://www.oecd-ilibrary.org/docserver/2e278150-en.pdf?expires=1684155083&id=id&accname=guest&checksum=5D6FCEF4C8BC7F10443E740A1832DA0C
    Orsolya|OECD/The Strategic and Responsible Use of Artificial Intelligence in the Public Sector of Latin America and the Caribbean/24/0|2022|https://www.oecd-ilibrary.org/deliver/1f334543-en.pdf?itemId=/content/publication/1f334543-en&mimeType=pdf
    Orsolya|OECD/The impact of AI on the workplace Main findings from the OECD AI surveys of employers and workers/23/0|2023|https://www.oecd-ilibrary.org/docserver/ea0a0fe1-en.pdf?expires=1684155576&id=id&accname=guest&checksum=B08290C7089EC3898A797F42E48752A1
    Orsolya|OECD/Tools for Trustworthy AI a Framework to compare Implementation Tools for Trustworthy AI Systems/25/0|2021|https://www.oecd-ilibrary.org/docserver/008232ec-en.pdf?expires=1684155305&id=id&accname=guest&checksum=40A2CB6572D2BD56588CCB7875747314
    Orsolya|OECD/Trustworthy artificial intelligence (AI) in education Promises and challenges/26/0|2020|https://www.oecd-ilibrary.org/docserver/a6c90fa9-en.pdf?expires=1684155444&id=id&accname=guest&checksum=A435CF63FD1F1C1548F6CA5BD4E97D15
    Orsolya|OECD/Venture Capital Investments in Artificial Intelligence Analysing Trends in VC in AI Companies from 2012 through 2020/27/0|2021|https://www.oecd-ilibrary.org/docserver/f97beae7-en.pdf?expires=1684155625&id=id&accname=guest&checksum=90745719FACC67F4FD8F95FA340A9ABF
    Orsolya|OECD/What are the OECD Principles on AI/28/0|2020|https://www.oecd-ilibrary.org/docserver/6ff2a1c4-en.pdf?expires=1684155213&id=id&accname=guest&checksum=47DD8B6DF2D9129E96D568E8A575B941
    Orsolya|OECD/Who develops AI-Related Innovations, Goods and Services A Firm-Level Analysis/29/0|2021|https://www.oecd-ilibrary.org/docserver/3e4aedd4-en.pdf?expires=1684155135&id=id&accname=guest&checksum=5AF7BB5C54110469A7BB41C5E9BEEE7C

    """
    )

    gr.Markdown("This application uses GPT to search the docs for relevant info and answer questions.")

    gr.HTML("<center> Powered by <a href='https://github.com/jerpint/buster'>Buster</a> 🤖</center>")

    completion = gr.State()

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
