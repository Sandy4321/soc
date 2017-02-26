from __future__ import absolute_import
from __future__ import print_function

import os
import pytest

from soc.modules import Nietzsche

sample_len = 7
num_samples = 13
nietzsche = Nietzsche(sample_len=7,
                      num_samples=13,
                      include_next=True)
nietzsche.load_data()


def test_shape():
    assert nietzsche.input_shape == [(sample_len,)]
    output_shape = [(1, nietzsche.num_chars,)]
    assert nietzsche.output_shape == output_shape


@pytest.mark.long
def test_iterate_data():
    data = nietzsche.train_data
    assert len(data) == 2

    x_train, y_train = data[0][0], data[1][0]
    assert [x_train.shape[1:]] == nietzsche.input_shape
    assert [y_train.shape[1:]] == nietzsche.output_shape
