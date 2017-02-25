from __future__ import absolute_import
from __future__ import print_function

import pytest

from soc.modules import MNIST


def test_download():
    m = MNIST()


if __name__ == '__main__':
    pytest.main([__file__])
