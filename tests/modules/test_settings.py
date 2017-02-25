from __future__ import absolute_import

import os
import pytest

import soc.modules._settings as settings


def test_chunk_size():
    chunk_size = settings.get_setting('chunk_size')
    assert isinstance(chunk_size, int)


def test_get_module_subdir():
    fname = 'test.ext'
    fpath = settings.get_module_subdir('module')
    assert fpath.endswith('module')


def test_set_data_dir():
    with pytest.raises(ImportError):
        settings.set_setting('data_dir', '/tmp/not/a/directory')
