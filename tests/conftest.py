"""Shared test configuration."""
import pytest


@pytest.fixture(autouse=True)
def recorded_errors(monkeypatch):
    """Every error dialog in the app, recorded as (title, text) instead of shown.

    `pose3d.ui.guard.report_error` is the one place a slot or job failure
    becomes a QMessageBox, so replacing it here — for every test, asked for or
    not — is what guarantees that an unexpected exception in a UI test can
    never park a modal dialog in front of a CI job with nobody to click it.
    Ask for the fixture by name to assert on what the user would have read.
    """
    recorded: list[tuple[str, str]] = []
    try:
        from pose3d.ui import guard
    except ImportError:                    # a build without PySide6
        yield recorded
        return
    monkeypatch.setattr(
        guard, "report_error",
        lambda parent, title, text: recorded.append((title, text)))
    yield recorded


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
