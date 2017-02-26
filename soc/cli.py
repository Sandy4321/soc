"""cli.py

Defines the command-line interface for SOC.
"""

from __future__ import absolute_import

import click
settings = dict(help_option_names=['-h', '--help'])

# Initializes logger.
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from .modules.mnist import mnist as mnist_cli
from .modules.nietzsche import nietzsche as nietzsche_cli
from .modules.ask_reddit import ask_reddit as ask_reddit_cli


@click.group(options_metavar='',
             subcommand_metavar='<command>',
             context_settings=settings)
def cli():
    """
    SOC: Data management system.
    """

cli.add_command(mnist_cli)
cli.add_command(nietzsche_cli)
cli.add_command(ask_reddit_cli)
