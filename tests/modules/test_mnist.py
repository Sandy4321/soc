from __future__ import absolute_import
from __future__ import print_function

from decorators import mpl_test

import os
import pytest

from soc.modules import MNIST
mnist = MNIST()


def test_shape():
    assert mnist.shape == ([(28, 28)], [()])


@pytest.mark.long
def test_iterate_data():
    data = mnist.train_data
    assert len(data) == 2

    x_data, y_data = data
    assert isinstance(x_data, list)
    assert isinstance(y_data, list)

    print(x_data[0].shape, y_data[0].shape)

    assert [i.shape[1:] for i in x_data] == mnist.input_shape
    assert [i.shape[1:] for i in y_data] == mnist.output_shape


@mpl_test
def test_visualize():
    mnist.visualize()


if __name__ == '__main__':
    pytest.main([__file__])
