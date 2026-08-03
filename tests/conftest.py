"""Shared test configuration."""
import pytest


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
