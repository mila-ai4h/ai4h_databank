import pandas as pd
from buster.busterbot import BusterAnswer
from buster.completers.base import Completion

from src.feedback import Feedback, FeedbackForm


class MockValidator:
    def check_sources_used(self, completion: Completion) -> bool:
        return True


def test_read_write_feedbackform():
    ff = FeedbackForm(
        good_bad="good",
        extra_info="extra",
        relevant_answer="relevant",
        relevant_length="length",
        relevant_sources="sources",
        length_sources="length",
        timeliness_sources="timeliness",
    )

    ff_json = ff.to_json()
    ff_back = FeedbackForm.from_dict(ff_json)

    assert ff.version == ff_back.version
    assert ff.good_bad == ff_back.good_bad
    assert ff.extra_info == ff_back.extra_info
    assert ff.relevant_answer == ff_back.relevant_answer
    assert ff.relevant_length == ff_back.relevant_length
    assert ff.relevant_sources == ff_back.relevant_sources
    assert ff.length_sources == ff_back.length_sources
    assert ff.timeliness_sources == ff_back.timeliness_sources


def test_read_write_feedback():
    n_samples = 3
    b = BusterAnswer(
        user_input="This is my input",
        completion=Completion(error=False, completor="This is my completed answer"),
        validator=MockValidator(),
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
    )

    f = Feedback(
        session_id="session_id",
        user_responses=[b],
        feedback_form=FeedbackForm(
            good_bad="good",
            extra_info="extra",
            relevant_answer="relevant",
            relevant_length="length",
            relevant_sources="sources",
            length_sources="length",
            timeliness_sources="timeliness",
        ),
        time="time",
    )

    f_json = f.to_json()
    f_json["_id"] = "0123"  # This is created by mongodb
    f_back = Feedback.from_dict(f_json)

    assert f.version == f_back.version
    assert f.session_id == f_back.session_id
    assert f.time == f_back.time
    assert f.feedback_form.version == f_back.feedback_form.version
    assert f.feedback_form.good_bad == f_back.feedback_form.good_bad
    assert f.feedback_form.extra_info == f_back.feedback_form.extra_info
    assert f.feedback_form.relevant_answer == f_back.feedback_form.relevant_answer
    assert f.feedback_form.relevant_length == f_back.feedback_form.relevant_length
    assert f.feedback_form.relevant_sources == f_back.feedback_form.relevant_sources
    assert f.feedback_form.length_sources == f_back.feedback_form.length_sources
    assert f.feedback_form.timeliness_sources == f_back.feedback_form.timeliness_sources
    assert len(f.user_responses) == len(f_back.user_responses)
    assert f.user_responses[0].version == f_back.user_responses[0].version
    assert f.user_responses[0].user_input == f_back.user_responses[0].user_input
    assert f.user_responses[0].completion.error == f_back.user_responses[0].completion.error
    assert f.user_responses[0].completion.text == f_back.user_responses[0].completion.text
    assert f.user_responses[0].completion.version == f_back.user_responses[0].completion.version
    assert f.user_responses[0].documents_relevant == f_back.user_responses[0].documents_relevant
    for col in f_back.user_responses[0].matched_documents.columns.tolist():
        assert col in f.user_responses[0].matched_documents.columns.tolist()
        assert (
            f.user_responses[0].matched_documents[col].tolist()
            == f_back.user_responses[0].matched_documents[col].tolist()
        )
