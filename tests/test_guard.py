"""One sink for every failure that happens inside a slot.

Qt swallows an exception raised inside a slot: the button appears to do
nothing, the traceback goes to a log nobody is looking at, and the app carries
on in whatever half-finished state the failure left. Only `_on_import` guarded
itself, with its own bespoke QMessageBox.

Two things follow from having exactly one sink. The user always gets the same
dialog, naming the log they have to send us. And the test suite can replace
that one function — see the autouse `recorded_errors` fixture in
tests/conftest.py — so an unexpected exception in a UI test can never park a
modal dialog in front of a CI job that has no one to click it.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from pose3d.ui.guard import guarded, report_error       # noqa: E402


class Panel:
    """Stands in for a widget with slots; `guarded` must not need a QWidget."""

    def __init__(self):
        self.ran = []

    @guarded
    def works(self, x):
        self.ran.append(x)
        return x * 2

    @guarded
    def explodes(self):
        raise ValueError("the rig went missing")


def test_a_healthy_slot_is_untouched(recorded_errors):
    p = Panel()
    assert p.works(21) == 42
    assert p.ran == [21]
    assert recorded_errors == []


def test_a_raising_slot_does_not_propagate(recorded_errors):
    p = Panel()
    assert p.explodes() is None          # Qt would have swallowed it anyway


def test_the_sink_records_the_type_and_the_log_path(recorded_errors):
    from pose3d.runtime import log_path

    Panel().explodes()

    assert len(recorded_errors) == 1
    title, text = recorded_errors[0]
    assert title
    assert "ValueError" in text
    assert "the rig went missing" in text
    assert str(log_path()) in text


def test_the_slot_name_is_in_the_message(recorded_errors):
    """"It failed" is not a report; which action failed is."""
    Panel().explodes()
    assert "explodes" in recorded_errors[0][1]


def test_the_decorator_keeps_the_functions_identity():
    assert Panel.explodes.__name__ == "explodes"


def test_keyboard_interrupt_still_gets_through(recorded_errors):
    class P:
        @guarded
        def stop(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        P().stop()
    assert recorded_errors == []


def test_report_error_shows_a_message_box_and_nothing_else(monkeypatch):
    """Unpatched, `report_error` is one QMessageBox — that is what makes
    replacing this single function enough to protect the whole suite.

    `report_error` here is the object this module imported before the autouse
    fixture rebound the module attribute, i.e. the real one.
    """
    from PySide6.QtWidgets import QMessageBox

    seen = {}
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda *a: seen.setdefault("args", a)))
    report_error(None, "Title", "Body")
    assert seen["args"] == (None, "Title", "Body")


def test_the_fixture_is_autouse_so_no_test_can_open_a_modal():
    """A test that never asks for the fixture is protected all the same."""
    import pose3d.ui.guard as guard

    assert guard.report_error is not report_error, (
        "tests/conftest.py's autouse recorded_errors fixture is not in place")
