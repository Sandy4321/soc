"""mnist.py

Module for dealing with MNIST data.
"""

from __future__ import absolute_import

from ._base import Module

from six.moves import cPickle as pkl
import gzip

import click
import numpy as np

# Where the MNIST data is hosted.
_MNIST_URL = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'


class MNIST(Module):
    """Module for MNIST dataset.

    This is a utility for caching and loading the MNIST data.
    """

    def __init__(self,
                 one_hot_output=True,
                 file_name='mnist'):
        self._data = None
        self.one_hot_output = one_hot_output
        self._file_name = '%s.pkl.gz' % file_name
        super(MNIST, self).__init__()

    def load_data(self):
        """Loads the training and testing data."""

        mnist_path = self.get_file(self._file_name,
                                   _MNIST_URL,
                                   use_bar=True,
                                   download=False)

        with gzip.open(mnist_path, 'rb') as f:
            self._data = pkl.load(f)

    @property
    def train_data(self):
        """Returns the training data, loading it if necessary."""

        if self._data is None:
            self.load_data()
        x_train, y_train = self._data[0]

        if self.one_hot_output:
            y_train = np.eye(10)[y_train]
        else:
            y_train = np.expand_dims(y_train, -1)

        return [x_train], [y_train]

    @property
    def test_data(self):
        """Returns the testing data, loading it if necessary."""

        if self._data is None:
            self.load_data()
        x_test, y_test = self._data[1]

        if self.one_hot_output:
            y_test = np.eye(10)[y_test]
        else:
            y_test = np.expand_dims(y_test, -1)

        return [x_test], [y_test]

    @property
    def input_shape(self):
        """MNIST data input shape."""

        return [(28, 28)]

    @property
    def output_shape(self):
        """MNIST data output shape."""

        if self.one_hot_output:
            return [(10,)]
        else:
            return [(1,)]

    def visualize(self, width=3, height=2):
        """Produces a visualization of the MNIST data.

        Args:
            width: int, the number of images width-wise.
            height: int, the number of images height-wise.
        """

        import matplotlib.pyplot as plt

        x_data, y_data = self.iterate_data(width * height, mode='test').next()
        x_data, y_data = x_data[0], y_data[0]

        plt.figure()
        for i in range(width * height):
            fig = plt.subplot(height, width, i + 1)
            fig.imshow(-x_data[i], cmap='gray')
            plt.title(y_data[i])
            fig.axis('off')
        plt.show()


@click.group()
def mnist():
    """MNIST command-line interface."""


@mnist.command()
@click.option('--file_name', default='mnist')
@click.option('--use_bar/--no_use_bar', default=True)
def download(file_name, use_bar):
    file_name = '%s.pkl.gz' % file_name
    MNIST().get_file(file_name, _MNIST_URL, use_bar=use_bar, download=True)


@mnist.command()
@click.option('--one_hot_output/--no_one_hot_output', default=True)
def shape(one_hot_output):
    m = MNIST(one_hot_output=one_hot_output)
    click.echo('Input shape: %s -- Output shape: %s'
               % (m.input_shape, m.output_shape))
