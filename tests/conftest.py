"""Shared test configuration."""
import pytest


@pytest.fixture(autouse=True)
def recorded_errors(monkeypatch):
    """Every user-facing failure, as (title, text), instead of a message box.

    `pose3d.ui.guard.report_error` is the ONE place a slot or a job failure
    becomes a QMessageBox, so replacing it here both keeps a modal dialog out
    of the headless suite and makes "did the user get told?" an assertion.
    """
    from pose3d.ui import guard
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        guard, "report_error",
        lambda parent, title, text: errors.append((title, text)))
    yield errors


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gate_missing(what, env): a required dependency is absent; see tests/gates.py")


def pytest_runtest_setup(item):
    """Fail — loudly, with the reason — rather than skip when CI said the
    dependency must be present. See tests/gates.py."""
    for m in item.iter_markers("gate_missing"):
        pytest.fail(f"{m.kwargs['what']}, but {m.kwargs['env']}=1 requires it",
                    pytrace=False)
