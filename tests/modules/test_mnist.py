from __future__ import absolute_import
from __future__ import print_function

from decorators import download_test, mpl_test

import os
import pytest

from soc.modules import MNIST
mnist = MNIST()


@download_test
def test_iterate_data():
    pass


@download_test
@mpl_test
def test_visualize():
    mnist.visualize()


if __name__ == '__main__':
    pytest.main([__file__])
