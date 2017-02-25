"""mnist.py

Module for dealing with MNIST data.
"""

from __future__ import absolute_import

from ._base import ArrayModule

from six.moves import cPickle as pkl
import gzip

# Where the MNIST data is hosted.
_MNIST_PATH = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'


class MNIST(ArrayModule):
    """Module for MNIST dataset.

    This loads the MNIST dataset from
    """

    def get_data(self):
        """Gets the MNIST training data.

        Returns:
            tuple of lists ([x_train], [y_train]), the inputs and outputs of
            the MNIST training data.
        """

        mnist_path = self.get_file('mnist.npz',
                                   _MNIST_PATH,
                                   use_bar=True,
                                   download=False)

        with gzip.open(mnist_path, 'rb') as f:
            (x_train, y_train), _ = pkl.load(f)

        return [x_train], [y_train]

    def get_test_data(self):
        """Gets the MNIST testing data.

        Returns:
            tuple of lists ([x_test], [y_test]), the inputs and outputs of
            the MNIST testing data.
        """

        mnist_path = self.get_file('mnist.npz',
                                   _MNIST_PATH,
                                   use_bar=True,
                                   download=False)

        with gzip.open(mnist_path, 'rb') as f:
            _, (x_test, y_test) = pkl.load(f)

        return [x_test], [y_test]

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
            fig.imshow(x_data[i])
            plt.title(y_data[i])
            fig.axis('off')
        plt.show()
