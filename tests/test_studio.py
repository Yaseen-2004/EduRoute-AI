"""Headless test of the evaluation studio page (no models needed)."""

import shutil

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from config import settings

RATER = "pytestrater"
GRADING = settings.BASE_DIR / "evaluation" / "results" / "grading"
LABELS = settings.BASE_DIR / "evaluation" / "results" / "claim_labelling"
REVIEW = settings.BASE_DIR / "config" / "branches_review"
CRITERIA = ("correctness", "completeness", "relevance", "clarity")


@pytest.fixture
def cleanup():
    existing = {p.name for p in REVIEW.glob("*.csv")} if REVIEW.exists() else set()
    yield
    for folder in (GRADING, LABELS):
        for f in folder.glob(f"*__{RATER}.csv"):
            f.unlink()
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()
    if REVIEW.exists():
        for f in REVIEW.glob("*.csv"):
            if f.name not in existing:
                f.unlink()
        if not any(REVIEW.iterdir()):
            shutil.rmtree(REVIEW)


def test_studio_grades_labels_and_reviews(cleanup):
    at = AppTest.from_file("../pages/2_Evaluation_Studio.py", default_timeout=60)
    at.run()
    assert not at.exception
    at.text_input(key="rater").input(RATER).run()
    assert not at.exception

    # Grade the first answer: all four criteria, then save.
    controls = [c for c in at.segmented_control if c.key and c.key.split("_")[0] in CRITERIA]
    assert len(controls) == 4
    for control in controls:
        control.set_value(2)
    next(b for b in at.button if b.label == "Save and next").click().run()
    assert not at.exception
    graded = pd.read_csv(next(GRADING.glob(f"*__{RATER}.csv")), dtype=str, keep_default_na=False)
    assert (graded["correctness_0_3"] == "2").sum() == 1 and (graded["clarity_0_3"] == "2").sum() == 1

    # Label one claim.
    next(b for b in at.button if b.label == "Supported").click().run()
    labelled = pd.read_csv(next(LABELS.glob(f"*__{RATER}.csv")), dtype=str, keep_default_na=False)
    assert (labelled["human_label"] == "supported").sum() == 1

    # Save a syllabus review for the first branch.
    next(b for b in at.button if b.label == "Save review").click().run()
    assert not at.exception
    assert any(REVIEW.glob("*.csv"))
