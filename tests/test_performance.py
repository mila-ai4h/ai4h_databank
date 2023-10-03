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
    result = detect_according_to_the_documentation("", answers)
    fail = 2
    total = 3
    assert (
        result
        == f"# Expressions Detector\n\nThis detector checks whether the system used expressions we want to discourage.\n- **According to the documentation**: {fail} / {total} ({fail / total * 100:04.2f} %)\n    - Include also the following variants: 'based on the documentation', 'the provided documents'\n\n\n"
    )


def test_evaluate_performance(monkeypatch, busterbot):
    questions = pd.DataFrame(
        [["What is the least relevant key pillar of AI policy in Canada?", True, False, "relevant", 0, True]],
        columns=["question", "valid_question", "valid_answer", "question_type", "group", "is_original"],
    )

    # Patch csv file to use
    monkeypatch.setattr(pd, "read_csv", lambda path: questions)

    evaluate_performance(busterbot)
