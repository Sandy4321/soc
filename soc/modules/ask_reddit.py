"""ask_reddit.py

Module for scraping data from AskReddit.
"""

from __future__ import print_function

from ._base import TextModule

import click
import gzip
import json
import time
import random

from six.moves import cPickle as pkl
from urllib2 import urlopen


class AskReddit(TextModule):
    """Module for querying and caching AskReddit results."""

    def __init__(self,
                 fname='ask_reddit',
                 max_question_len=100,
                 max_answer_len=100,
                 one_hot_input=False,
                 one_hot_output=True,
                 **kwargs):
        """Creates an AskReddit Module object.

        Args:
            max_question_len: int, the maximum question length, in characters.
            max_answer_len: int, the maximum answer length, in characters.
        """

        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.fname = fname + '.pkl.gz'
        self._data = None
        self.one_hot_input = one_hot_input
        self.one_hot_output = one_hot_output

        kwargs['level'] = 'word'
        super(AskReddit, self).__init__(**kwargs)

    def load_data(self):
        """Loads the training and testing data."""

        fpath = self.get_path(self.fname)

        if not os.path.exists(fpath):
            raise RuntimeError('No file found at "%s". Use the command-line '
                               'interface to download data.' % fpath)

        with gzip.open(fpath, 'rb') as f:
            self._data = pkl.load(f)

    @property
    def train_data(self):
        """Returns the training data, loading it if necessary."""

        if self._data is None:
            self.load_data()

        questions, answers = self._data
        questions = self.encode(questions,
                                max_len=self.max_question_len,
                                update_dicts=True,
                                one_hot=self.one_hot_input)
        answers = self.encode(answers,
                              max_len=self.max_answer_len,
                              update_dicts=True,
                              one_hot=self.one_hot_output)

        return [questions], [answers]

    @property
    def input_shape(self):
        """Gets the input shape as a list of tuples."""

        if self.one_hot_input:
            return [(self.max_question_len, self.num_chars)]
        else:
            return [(self.max_question_len,)]

    @property
    def output_shape(self):
        """Gets the output shape as a list of tuples."""

        if self.one_hot_output:
            return [(self.max_answer_len, self.num_chars)]
        else:
            return [(self.max_answer_len,)]


@click.group()
def ask_reddit():
    """AskReddit command-line interface."""


@ask_reddit.command()
@click.option('--fname', default='askreddit')
@click.option('--num_results', default=1000)
@click.option('--override', default=False)
@click.option('--num_comments', default=5)
@click.option('--time_filter',
              type=click.Choice(['hour', 'day', 'week',
                                 'month', 'year', 'all']),
              default='all')
@click.option('--wait_time', default=0.5)
def download(fname,
             num_results,
             override,
             num_comments,
             time_filter,
             wait_time):

    # Uses the Reddit API wrapper.
    import praw

    # Assumes credentials are set in environment variables.
    reddit = praw.Reddit()

    # Gets the save path.
    fname = fname + '.pkl.gz'
    fpath = AskReddit().get_path(fname)

    if override and os.path.exists(fpath):
        raise ValueError('A file already exists at "%s". Use the --override '
                         'flag to get rid of it, or use a different file name.')

    questions = []
    answers = []

    bar = click.progressbar(length=num_results, label='ask_reddit')

    num_parsed = 0
    for submission in reddit.subreddit('AskReddit').top(time_filter):
        title = submission.title.strip()

        # Filters out non-question threads.
        if not title.endswith('?'):
            continue

        for comment in submission.comments[:num_comments]:
            questions.append(title)
            answers.append(comment.body)
            num_parsed += 1
            bar.update(num_parsed)

            if num_parsed >= num_results:
                break

        if num_parsed >= num_results:
            break

    bar.finish()

    # Saves the output.
    with gzip.open(fpath, 'wb') as f:
        pkl.dump([questions, answers], f)
    click.echo('Done')


@ask_reddit.command()
@click.option('--max_question_len', default=100)
@click.option('--max_answer_len', default=100)
@click.option('--one_hot_input/--no_one_hot_input', default=False)
@click.option('--one_hot_output/--no_one_hot_output', default=True)
def shape(max_question_len,
          max_answer_len,
          one_hot_input,
          one_hot_output):
    n = AskReddit(max_question_len=max_question_len,
                  max_answer_len=max_answer_len,
                  one_hot_input=one_hot_input,
                  one_hot_output=one_hot_output)
    click.echo('Input shape: %s -- Output shape: %s'
               % (n.input_shape, n.output_shape))
