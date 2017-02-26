from __future__ import absolute_import
from __future__ import print_function

import os
import pytest

from soc.modules import AskReddit
ask_reddit = AskReddit(max_question_len=100,
                       max_answer_len=100)


def test_shape():
    assert ask_reddit.shape == ([(100,)], [(100, 1)])


if __name__ == '__main__':
    pytest.main([__file__])
