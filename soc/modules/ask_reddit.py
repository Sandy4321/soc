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
                 max_question_len=100,
                 max_answer_len=100,
                 **kwargs):
        """Creates an AskReddit Module object.

        Args:
            max_question_len: int, the maximum question length, in characters.
            max_answer_len: int, the maximum answer length, in characters.
        """

        super(AskReddit, self).__init__()


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

    with click.progressbar(length=num_results) as bar:
        num_parsed = 0
        while num_parsed < num_results:
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

    # Saves the output.
    pkl.dump([questions, answers], gzip.open(fpath, 'wb'))
    click.echo('Saved to "%s"' % fpath)
