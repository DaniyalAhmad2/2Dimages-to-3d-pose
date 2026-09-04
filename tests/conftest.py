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


@pytest.fixture(autouse=True)
def closed_windows():
    """Destroy each test's windows while Qt is still able to do it properly.

    `QOpenGLWidget` — which the 3D view is — crashes inside its own destructor
    when Python's garbage collector gets to it at interpreter exit: by then the
    cycle collector is tearing objects down in its own order, with no
    QApplication left to unwind the context against. It is not a fault in the
    app (which closes its windows while Qt is alive), but it dumps core after
    the suite's last green line, and a test run that ends in "Segmentation
    fault" is indistinguishable from the app doing it.

    So every test hands its windows back the way the app does: close, then
    `deleteLater`, then flush the deferred deletes.
    """
    yield
    try:
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
    except ImportError:                    # a build without PySide6
        return
    app = QApplication.instance()
    if app is None:
        return
    for widget in list(app.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


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
