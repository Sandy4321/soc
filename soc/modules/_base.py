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
import warnings
import six
import re

from urllib2 import urlopen

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

    @staticmethod
    def validate_dataset(x_data, y_data):
        """Validates properties about the dataset.

        Args:
            x_data: list of lists or Numpy arrays, the input data.
            y_data: list of lists or Numpy arrays, the output / target data.

        Raises:
            ValueError: if there is a validation problem.
        """

        # Checks that the batch sizes are equal.
        def _check_batch_dim(args):
            return not args or [len(i) == args[0] for i in args[1:]]

        if not _check_batch_dim(x_data):
            raise ValueError('The batch dimension of x_data is not equal '
                             'for all arrays in the dataset. Got: %s'
                             % str([len(i) for i in x_data]))

        if not _check_batch_dim(y_data):
            raise ValueError('The batch dimension of y_data is not equal '
                             'for all arrays in the dataset. Got: %s'
                             % str([len(i) for i in y_data]))

        if x_data and y_data and not len(x_data[0]) == len(y_data[0]):
            raise ValueError('Different number of x_data and y_data samples: '
                             'x_data has %d, y_data has %d'
                             % (len(x_data[0]), len(y_data[0])))

        return True

    @property
    def shape(self):
        """Gets a tuple of (input_shape, output_shape)."""

        return self.input_shape, self.output_shape

    def __call__(self):
        return self.train_data

    def load_data(self):
        """Method signature for doing all the preprocessing."""

        raise NotImplementedError()

    @property
    def test_data(self):
        """Gets the testing data, a tuple (x_test, y_test)."""

        # return [np.array()], [np.array()]
        raise NotImplementedError()

    @property
    def train_data(self):
        """Gets the training data, a tuple (x_train, y_train)."""

        # return [np.array()], [np.array()]
        raise NotImplementedError()

    @property
    def input_shape(self):
        """Gets the input shape as a list of tuples."""

        # return [()]
        raise NotImplementedError()

    @property
    def output_shape(self):
        """Gets the output shape as a list of tuples."""

        # return [()]
        raise NotImplementedError()

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

        if mode not in ('train', 'test'):
            raise ValueError('Invalid mode: "%s" (should be "train" or '
                             '"test")' % mode)

        while True:
            # Updates the data.
            if mode == 'train':
                x_data, y_data = self.train_data
            else:
                x_data, y_data = self.test_data

            # Gets the number of samples.
            num_samples = x_data[0].shape[0]

            if randomize:
                idxs = np.arange(num_samples)

            if randomize:
                np.random.shuffle(idxs)
                for i in range(batch_size, num_samples, batch_size):
                    idx = idxs[i - batch_size:i]
                    yield [x[idx] for x in x_data], [y[idx] for y in y_data]
            else:
                for i in range(batch_size, num_samples, batch_size):
                    yield x_data[i - batch_size:i], y_data[i - batch_size:i]


class TextModule(Module):
    """Defines a module where data are strings.

    This module should have its data as a string.
    """

    def __init__(self, level='char', missing='?', end='|'):
        self.missing = missing
        self.end = end
        self._char_to_idx = {end: 0}
        self._idx_to_char = {0: end}

        if level == 'char':
            self.serialize = None
        elif level == 'word':
            regex_str = '|'.join(['[a-zA-Z]+', '\d+', '[\?\.\!\(\)]'])
            self.serialize = lambda x: re.findall(regex_str, x)
        else:
            raise ValueError('"level" should be one of ["char", "word"], got '
                             '"%s"' % level)

        super(TextModule, self).__init__()

    def update_dicts_with_str(self, string):
        """Adds a string to the look-up dictionaries.

        Args:
            string: str, the string to add.
        """

        if self.serialize is None:
            string_list = set(string)
        else:
            string_list = set(self.serialize(string))

        for token in string_list:
            self.update_dicts(token)

    def update_dicts(self, c):
        """Adds a character to the look-up dictionaries.

        Args:
            c: str, the character to add to the dictionary.

        Returns:
            bool, False if the character was already in the dictionary.
        """

        if c in self._char_to_idx:
            return False

        idx = self.num_chars
        self._char_to_idx[c] = idx
        self._idx_to_char[idx] = c

        return True

    def decode(self, data, argmax=True):
        """Decodes a Numpy array to a string or list of strings.

        Args:
            data: int or Numpy array, the data to decode.
            argmax: bool, whether or not to take the argmax over the last
                dimension in the data array.

        Returns:
            text: string or list of strings, the decoded text.
        """

        if argmax:
            data = np.argmax(data, axis=-1)

        if isinstance(data, int):
            text = self._idx_to_char.get(data, self.missing)
        elif np.ndim(data) == 1:
            if self.serialize is None:
                text =  ''.join(self._idx_to_char.get(x, self.missing)
                                for x in data if x > 0)
            else:
                text =  ' '.join(self._idx_to_char.get(x, self.missing)
                                 for x in data if x > 0)
        elif np.ndim(data) == 2:
            text = [self.decode(x, argmax=False) for x in data]
        else:
            raise ValueError('Invalid number of dimensions: %d. The provided '
                             'array should be 1 dimension or 2 dimensions '
                             'after doing the argmax over the last dimension.'
                             % np.ndim(data))

        return text

    @property
    def num_chars(self):
        """Returns the number of characters in the dictionary."""

        return len(self._idx_to_char)

    def encode(self, data, max_len, update_dicts=False, one_hot=False):
        """Encodes a string or list of strings to a Numpy array.

        Args:
            data: string or list of strings, the data to encode.
            max_len: int, maximum length of a string.
            update_dicts: bool, if set, updates the dictionary while encoding
                the strings.
            one_hot: bool, if set, return

        Returns:
            arr: the Numpy array, with shape (max_len) if the data is a string
                or (len(data), max_len) if the data is a list of strings.
        """

        if one_hot and update_dicts:
            raise ValueError('one_hot and update_dicts cannot both be set.')

        if isinstance(data, (list, tuple)):
            arr = np.stack([self.encode(string, max_len,
                                        update_dicts=update_dicts,
                                        one_hot=one_hot)
                            for string in data])
        elif isinstance(data, six.string_types):
            if self.serialize is not None:
                data = self.serialize(data)

            if update_dicts:
                new_chars = [c for c in data if c not in self._char_to_idx]
                for c in new_chars:
                    self.update_dicts(c)

            try:
                if one_hot:
                    eye = np.eye(self.num_chars)
                    arr = np.zeros(shape=(max_len, self.num_chars))
                    data = np.asarray([eye[self._char_to_idx[c]] for c in data])
                    arr[:len(data)] = data[:max_len]
                else:
                    arr = np.zeros(shape=(max_len,))
                    data = np.asarray([self._char_to_idx[c] for c in data])
                    arr[:len(data)] = data[:max_len]

            except KeyError:
                raise KeyError('You tried to encode a character that wasn\'t '
                               'in the look-up dict. Setting update_dict=True '
                               'will update the look-up dict as the characters '
                               'are encoded.')
        else:
            raise ValueError('The data must be either a string or list of '
                             'strings. Got "%s"' % (data))

        return arr

    @staticmethod
    def get_string_samples(string, sample_len, num_samples, include_next=False):
        """Returns num_samples substrings from the big string.

        Args:
            string: str, the string to draw samples from.
            sample_len: int, the length of each sample.
            num_samples: int, the number of samples to generate.

        Returns:
            (x_data, y_data) if include_next, otherwise just x_data.
        """

        min_length = sample_len + num_samples - 1
        if include_next:
            min_length += 1

        if len(string) < min_length:
            raise ValueError('The string to draw samples from is too short. '
                             'It is only %d characters, but it should be at '
                             'least %d characters' (len(string), min_length))

        idxs = np.random.choice(len(string) - sample_len, num_samples)
        x_data = [string[i:i + sample_len] for i in idxs]

        if include_next:
            y_data = [string[i + sample_len:i + sample_len + 1] for i in idxs]
            return x_data, y_data
        else:
            return x_data
