#!/usr/bin/env python

from setuptools import setup, find_packages

version = '0.0.1'
github_url = 'https://github.com/codekansas/soc'

setup(
    name='soc',
    version=version,
    description='Easily access online datasets.',
    author='codekansas',
    author_email='ben@bolte.cc',
    url=github_url,
    packages=find_packages(),

    # These are the open-source packages SOC is built around.
    install_requires=[
        'numpy',
        'six',
        'click',
    ],

    # For running PyTest: ./setup.py pytest
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    # Installation adds the command-line interface.
    entry_points = {"console_scripts": ['soc = soc.cli:cli']},

    long_description='See "%s"' % github_url,
    license='MIT'
)
