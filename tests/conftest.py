import pytest

def pytest_addoption(parser):
    parser.addoption('--mpl',
                     action='store_true',
                     help='if set, run matplotlib tests')
