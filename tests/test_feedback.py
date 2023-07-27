import pandas as pd
from buster.completers import Completion

from src.feedback import Feedback, FeedbackForm


class MockValidator:
    def check_sources_used(self, completion: Completion) -> bool:
        return True

    def check_answer_relevance(self, *args, **kwargs) -> bool:
        return True


def test_read_write_feedbackform():
    ff = FeedbackForm(relevant_answer="relevant", relevant_sources="sources", extra_info="extra info")

    ff_json = ff.to_json()
    ff_back = FeedbackForm.from_dict(ff_json)

    assert ff.extra_info == ff_back.extra_info
    assert ff.relevant_answer == ff_back.relevant_answer
    assert ff.relevant_sources == ff_back.relevant_sources


def test_read_write_feedback():
    n_samples = 3
    b = Completion(
        error=False,
        user_input="This is my input",
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

    f = Feedback(
        username="test user",
        user_responses=[b],
        feedback_form=FeedbackForm(
            extra_info="extra",
            relevant_answer="relevant",
            relevant_sources="sources",
        ),
        time="time",
    )

    f_json = f.to_json()
    f_json["_id"] = "0123"  # This is created by mongodb
    f_back = Feedback.from_dict(f_json, feedback_cls=FeedbackForm)

    assert f.username == f_back.username
    assert f.time == f_back.time
    assert f.feedback_form.extra_info == f_back.feedback_form.extra_info
    assert f.feedback_form.relevant_answer == f_back.feedback_form.relevant_answer
    assert f.feedback_form.relevant_sources == f_back.feedback_form.relevant_sources
    assert len(f.user_responses) == len(f_back.user_responses)
    assert f.user_responses[0].user_input == f_back.user_responses[0].user_input
    assert f.user_responses[0].error == f_back.user_responses[0].error
    assert f.user_responses[0].answer_text == f_back.user_responses[0].answer_text
    assert f.user_responses[0].answer_relevant == f_back.user_responses[0].answer_relevant
    assert f.user_responses[0].question_relevant == f_back.user_responses[0].question_relevant
    for col in f_back.user_responses[0].matched_documents.columns.tolist():
        assert col in f.user_responses[0].matched_documents.columns.tolist()
        assert (
            f.user_responses[0].matched_documents[col].tolist()
            == f_back.user_responses[0].matched_documents[col].tolist()
        )
