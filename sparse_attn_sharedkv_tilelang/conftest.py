"""Pytest config for the TileLang SparseAttnSharedKV suite.

Large-S1 cases are marked ``slow`` -- the CPU golden reference takes
minutes when ``S1`` is in the thousands. They are skipped unless
``--runslow`` is passed.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow cases (large S1; the CPU golden takes minutes)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: large-S1 case; skipped unless --runslow is given"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow case; pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
