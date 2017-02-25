"""_base.py

Defines the base template for a module. To add a new module, you should
override the Module class or one of its subclasses.
"""

from __future__ import absolute_import

from ._settings import get_setting, get_module_subdir

import click
import logging
import os
import uuid

from urllib2 import urlopen, Request, HTTPError

import numpy as np


class Module(object):
    """Defines the abstract module class."""

    def __init__(self):
        self.module_name = self.__class__.__name__.lower()
        self.data_subdir = get_module_subdir(self.module_name)

    def get_path(self, fname):
        """Returns the path to the specified module file.

        Args:
            fname: str, the name of the requested module file.

        Returns:
            fpath: str, the full path to the file.
        """

        fpath = os.path.join(self.data_subdir, fname)
        return fpath

    @staticmethod
    def get_unique(ext):
        """Returns a unique filename with the desired extension.

        Args:
            ext: str, the extension to use.

        Returns:
            a unique file path in this module with the right extension.
        """

        return '%s.%s' % (str(uuid.uuid4()), ext)

    def validate_file(self, fname, md5_hash):
        """Validates a file's MD5 hash, as done in Keras.

        Args:
            fpath: str, path to the file to validate.
            md5_hash: the md5 hash being validated against.

        Returns:
            is_valid: bool, True if the file is valid.
        """

        fpath = self.get_file_path(fname)

        hasher = hashlib.md5()
        with open(fpath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        is_valid = str(hasher.hexdigest()) == str(md5_hash)

        return is_valid


    def get_file(self, fname, url, use_bar=True, download=False):
        """Retrieves a file from the specified URL, or loads it if it exists.

        Args:
            fname: str, name of the file to download.
            url: str, original URL of the file.
            use_bar: bool, whether or not to use the progress bar.
            download: bool, if set, always download a new file.

        Returns:
            fpath: str, path to the downloaded file.
        """

        fpath = self.get_path(fname)

        if not os.path.exists(fpath) or download:
            response = urlopen(url)
            info, url = response.info(), response.url

            # Gets the total file size from the header.
            fsize = int(info.get('Content-Length').strip())

            if use_bar:
                bar = click.progressbar(length=fsize,
                                        label=self.module_name,
                                        fill_char='=',
                                        empty_char='.')

            # Downloads the file.
            chunk_size = get_setting('chunk_size')
            with open(fpath, 'wb') as f:
                chunk = response.read(chunk_size)
                while chunk:
                    f.write(chunk)
                    if use_bar:
                        bar.update(chunk_size)
                    chunk = response.read(chunk_size)

            if use_bar:
                bar.finish()

        return fpath


class ArrayModule(Module):
    """Defines a module where data are Numpy arrays.

    This module should have either two arrays (x_train, y_train) or four arrays
    ((x_train, y_train), (x_test, y_test)).
    """

    def get_data(self):
        """Gets the training data for this module.

        Returns:
            tuple of (input_data, output_data), where input_data and
            output_data are lists of Numpy arrays.
        """

        raise NotImplementedError()

    def get_test_data(self):
        """Gets the testing data for this module.

        Returns:
            tuple of (input_data, output_data), where input_data and
            output_data are lists of Numpy arrays.
        """

        raise NotImplementedError()

    def __call__(self):
        return self.get_data()

    @staticmethod
    def validate_dataset(x_data, y_data):
        """Validates properties about the dataset.

        Args:
            x_data: list of Numpy arrays, the input data.
            y_data: list of Numpy arrays, the output / target data.

        Raises:
            ValueError: if there is a validation problem.
        """

        # Checks that the batch sizes are equal.
        def _check_batch_dim(*args):
            return len(args) == 0 or [i == args[0] for i in args]

        if not _check_batch_dim(x_data):
            raise ValueError('The batch dimension of x_data is not equal '
                             'for all arrays in the dataset. Got: %s'
                             % str([i.shape[0] for i in x_data]))

        if not _check_batch_dim(y_data):
            raise ValueError('The batch dimension of y_data is not equal '
                             'for all arrays in the dataset. Got: %s'
                             % str([i.shape[0] for i in y_data]))

        return True


    def iterate_data(self,
                     batch_size,
                     mode='train',
                     randomize=True):
        """Iterates the training data.

        Args:
            batch_size: int, the size of each batch.
            mode: str, 'train' or 'test'.
            randomize: bool, whether to randomize the batch entries.

        Yields:
            tuple of lists (x_data, y_data), where x_data and y_data are lists
            of numpy arrays with first dimension batch_size.
        """

        if mode == 'train':
            x_data, y_data = self.get_data()
        elif mode == 'test':
            try:
                x_data, y_data = self.get_test_data()
            except NotImplementedError:
                x_data, y_data = self.get_data()

        # Validates whichever dataset is being used.
        ArrayModule.validate_dataset(x_data, y_data)

        # Gets the number of samples.
        nb_samples = x_data[0].shape[0]

        if not randomize:
            count = 0

        # Continually yield batches.
        while True:
            if randomize:
                idx = np.random.choice(nb_samples, batch_size)
            else:
                idx = [i % nb_samples for i in
                       range(count, count + batch_size)]
                count = (count + batch_size) % nb_samples
            x_batch = [x[idx] for x in x_data]
            y_batch = [y[idx] for y in y_data]
            yield x_batch, y_batch
