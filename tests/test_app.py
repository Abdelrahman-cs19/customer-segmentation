"""Smoke-test the Streamlit dashboard by actually executing it headlessly."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "final_model.joblib").exists(), reason="run the pipeline first")


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=90).run()


def test_dashboard_runs_and_has_all_tabs():
    at = _app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Segment profiles", "EDA", "Preprocessing", "Test the model", "Feature explorer",
                                            "Customer lookup", "Model comparison", "Downloads"]


def test_customer_lookup_works():
    at = _app()
    at.text_input[0].set_value("C10001").run()
    assert not at.exception
    assert any("C10001" in s.value for s in at.success)


def test_test_the_model_form_assigns_the_matching_segment():
    at = _app()
    names = at.selectbox[0].options  # preset selector: the 4 segments + blank
    seg = [n for n in names if not n.startswith("Blank")][0]
    at.selectbox[0].select(seg).run()
    next(b for b in at.button if b.label == "Assign segment").click().run()
    assert not at.exception
    assigned = [s.value for s in at.success if "Assigned segment" in s.value]
    assert assigned and seg in assigned[0], (seg, assigned)
