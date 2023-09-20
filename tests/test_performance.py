import pandas as pd

from performance.test_performance import (
    busterbot,
    detect_according_to_the_documentation,
    evaluate_performance,
)


def test_detect_according_to_the_documentation():
    answers = pd.DataFrame(
        {
            "answer_text": [
                "According to the documentation, this is a unit test",
                "I have a lot of information and can answer cool questions",
                "French have the best cheese, based on the provided documents.",
            ]
        }
    )
    fail, total = detect_according_to_the_documentation(answers)
    assert fail == 2
    assert total == 3


def test_evaluate_performance(monkeypatch, busterbot):
    questions = pd.DataFrame(
        [["What is the least relevant key pillar of AI policy in Canada?", True, False, "relevant"]],
        columns=["question", "valid_question", "valid_answer", "question_type"],
    )

    # Patch csv file to use
    monkeypatch.setattr(pd, "read_csv", lambda path: questions)

    evaluate_performance(busterbot)
