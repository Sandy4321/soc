from __future__ import absolute_import

import os
import pytest

import soc.modules._base as base

module = base.Module()


def test_get_path():
    fname = 'test.ext'
    fpath = module.get_path(fname)

    assert fpath.endswith(os.path.join('module', fname))


def test_get_unique():
    unique_name = base.Module.get_unique('txt')
    unique_path = module.get_path(unique_name)

    assert not os.path.exists(unique_path)
    assert unique_name.endswith('.txt')


@pytest.mark.long
def test_get_file():
    url = 'http://ben.bolte.cc/resume.pdf'
    fname = 'resume.pdf'
    fpath = module.get_file(fname, url, use_bar=True, download=True)
    fpath, fname_new = os.path.split(fpath)

    assert fname_new == fname
    assert fpath.endswith('module')


if __name__ == '__main__':
    pytest.main([__file__])
