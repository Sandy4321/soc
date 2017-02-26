"""nietzsche.py

Module for dealing with the Nietzsche text dataset.
"""

from __future__ import print_function

from ._base import TextModule

from six.moves import cPickle as pkl

# Where the Nietche data is hosted.
_NIETZSCHE_URL = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt'


class Nietzsche(TextModule):
    """Module for Nietzsche dataset.

    This is a utility for caching and loading the Nietzsche data.
    """

    def __init__(self,
                 sample_len,
                 num_samples,
                 num_test=100,
                 include_next=True,
                 one_hot_input=False,
                 one_hot_output=True,
                 **kwargs):
        """Creates a Nietzsche Module object.

        Args:
            sample_len: int, the length of samples to draw from the module.
            num_samples: int, the number of samples to load into memory at once.
            num_test: int, number of test samples.
            include_next: bool, if set, y_data is character right after the
                last character in x_data.
            one_hot_input: bool, whether or not to use one-hot encoding on the
                inputs.
            one_hot_output: bool, whether or not to use one-hot encoding on the
                outputs.
        """

        self.sample_len = sample_len
        self.num_samples = num_samples
        self.num_test = 100
        self.include_next = include_next
        self.one_hot_input = one_hot_input
        self.one_hot_output = one_hot_output
        self._text = None
        super(Nietzsche, self).__init__(**kwargs)

    def load_data(self):
        """Loads the training and testing data."""

        nietzsche_path = self.get_file('nietzsche.txt',
                                       _NIETZSCHE_URL,
                                       use_bar=True,
                                       download=False)

        with open(nietzsche_path, 'r') as f:
            text = f.read()

        # Add all the characters to the dictionary.
        for c in set(text):
            self.update_dicts(c)

        cut_idx = self.num_test + self.sample_len
        self._text = (text[:cut_idx], text[cut_idx:])

    def _process_text(self, text):
        """Convenience method for train_data and test_data."""

        data = TextModule.get_string_samples(text,
                                             self.sample_len,
                                             self.num_samples,
                                             include_next=self.include_next)

        if self.include_next:
            x_train, y_train = data
            x_train = self.encode(x_train,
                                  max_len=self.sample_len,
                                  update_dicts=False,
                                  one_hot=self.one_hot_input)
            y_train = self.encode(y_train,
                                  max_len=1,
                                  update_dicts=False,
                                  one_hot=self.one_hot_output)
            return [x_train], [y_train]
        else:
            x_train = self.encode(x_train,
                                  max_len=self.sample_len,
                                  update_dicts=False,
                                  one_hot=self.one_hot_input)
            return [x_train], []

    @property
    def train_data(self):
        """Returns the training data, loading it if necessary."""

        if self._text is None:
            self.load_data()
        return self._process_text(self._text[0])

    @property
    def test_data(self):
        """Returns the testing data, loading it if necessary."""

        if self._text is None:
            self.load_data()
        return self._process_text(self._text[1])

    @property
    def input_shape(self):
        """Gets the input shape as a list of tuples."""

        if self.one_hot_input:
            return [(self.sample_len, self.num_chars)]
        else:
            return [(self.sample_len,)]

    @property
    def output_shape(self):
        """Gets the output shape as a list of tuples."""

        if self.include_next:
            if self.one_hot_output:
                return [(1, self.num_chars)]
            else:
                return [(1,)]
        else:
            return [()]
