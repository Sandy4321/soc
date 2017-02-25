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
    unique_path = module.get_unique('txt')
    fpath, fname = os.path.split(unique_path)

    assert not os.path.exists(unique_path)
    assert fname.endswith('.txt')
    assert fpath.endswith('module')


def test_not_implemented():
    with pytest.raises(NotImplementedError):
        module.download()

    with pytest.raises(NotImplementedError):
        module.pd()


if __name__ == '__main__':
    pytest.main([__file__])
