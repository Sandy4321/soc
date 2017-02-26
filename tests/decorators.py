from __future__ import absolute_import

import pytest

mpl_test = pytest.mark.skipif(
    not pytest.config.getoption("--mpl"),
    reason="matplotlib test; need --mpl option to run"
)
