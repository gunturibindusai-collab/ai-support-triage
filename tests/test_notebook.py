"""
Smoke test for the ticket triage notebook.

Since all project code lives in a single Jupyter notebook (not separate .py
modules), we can't import individual functions to test them in isolation.
Instead, this test runs the ENTIRE notebook top to bottom, exactly like
`make run` does, and checks two things:

1. No cell raised an error during execution.
2. The final pipeline output actually printed the fields we expect
   (predicted_queue, retrieved_resolutions, draft_reply, escalation_decision).

Run with: pytest tests/ -v
NOTE: this executes real LLM calls through OpenRouter, so it needs a valid
.env with OPENROUTER_API_KEY, and will take a few minutes to run (it re-runs
the classifier training and embedding steps too).
"""

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK_PATH = "notebooks/001_eda.ipynb"


def _execute_notebook():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=1200, kernel_name="python3")
    client.execute()
    return nb


def _get_all_output_text(nb):
    text_chunks = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if "text" in output:
                text_chunks.append(output["text"])
            if "data" in output and "text/plain" in output["data"]:
                text_chunks.append(output["data"]["text/plain"])
    return "\n".join(text_chunks)


def test_notebook_runs_without_errors():
    try:
        nb = _execute_notebook()
    except CellExecutionError as e:
        raise AssertionError(f"Notebook execution failed: {e}")

    assert nb is not None


def test_notebook_produces_expected_pipeline_output():
    nb = _execute_notebook()
    all_text = _get_all_output_text(nb)

    expected_markers = [
        "predicted_queue",
        "retrieved_resolutions",
        "draft_reply",
        "escalation_decision",
    ]

    for marker in expected_markers:
        assert marker in all_text, f"Expected '{marker}' to appear in notebook output, but it didn't."
