import pandas as pd

from buster.completers import Completion
from buster.utils import UserInputs
from src.feedback import FeedbackForm, Interaction


class MockValidator:
    def check_sources_used(self, completion: Completion) -> bool:
        return True

    def check_answer_relevance(self, *args, **kwargs) -> bool:
        return True


def test_read_write_feedbackform():
    ff = FeedbackForm(
        overall_experience="yes",
        clear_answer="yes",
        accurate_answer="yes",
        relevant_sources="yes",
        relevant_sources_selection=["source 1", "source 2"],
        relevant_sources_order="yes",
        expertise="Beginner",
        extra_info="Helpful",
    )

    ff_json = ff.to_json()
    ff_back = FeedbackForm.from_dict(ff_json)

    assert ff.overall_experience == ff_back.overall_experience
    assert ff.clear_answer == ff_back.clear_answer
    assert ff.accurate_answer == ff_back.accurate_answer
    assert ff.relevant_sources == ff_back.relevant_sources
    assert ff.relevant_sources_order == ff_back.relevant_sources_order
    assert ff.relevant_sources_selection == ff_back.relevant_sources_selection
    assert ff.expertise == ff_back.expertise
    assert ff.extra_info == ff_back.extra_info


def test_read_write_feedback():
    n_samples = 3
    b = Completion(
        error=False,
        user_inputs=UserInputs(original_input="This is my input"),
        answer_text="This is my completed answer",
        matched_documents=pd.DataFrame.from_dict(
            {
                "title": ["test"] * n_samples,
                "url": ["http://url.com"] * n_samples,
                "content": ["cool text"] * n_samples,
                "embedding": [[0.0] * 1000] * n_samples,
                "n_tokens": [10] * n_samples,
                "source": ["fake source"] * n_samples,
            }
        ),
        validator=MockValidator(),
    )

    f = Interaction(
        username="test user",
        session_id="test id",
        user_completions=[b],
        form=FeedbackForm(
            overall_experience="yes",
            clear_answer="yes",
            accurate_answer="yes",
            relevant_sources="yes",
            relevant_sources_selection=["source 1", "source 2"],
            relevant_sources_order="yes",
            expertise="Beginner",
            extra_info="Helpful",
        ),
        time="time",
    )

    f_json = f.to_json()
    f_json["_id"] = "0123"  # This is created by mongodb
    f_back = Interaction.from_dict(f_json, feedback_cls=FeedbackForm)

    assert f.username == f_back.username
    assert f.time == f_back.time
    assert f.form.extra_info == f_back.form.extra_info
    assert f.form.relevant_sources == f_back.form.relevant_sources
    assert len(f.user_completions) == len(f_back.user_completions)
    assert f.user_completions[0].user_inputs == f_back.user_completions[0].user_inputs
    assert f.user_completions[0].error == f_back.user_completions[0].error
    assert f.user_completions[0].answer_text == f_back.user_completions[0].answer_text
    assert f.user_completions[0].answer_relevant == f_back.user_completions[0].answer_relevant
    assert f.user_completions[0].question_relevant == f_back.user_completions[0].question_relevant
    for col in f_back.user_completions[0].matched_documents.columns.tolist():
        assert col in f.user_completions[0].matched_documents.columns.tolist()
        assert (
            f.user_completions[0].matched_documents[col].tolist()
            == f_back.user_completions[0].matched_documents[col].tolist()
        )
