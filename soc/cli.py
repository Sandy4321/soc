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

from .modules import MNIST


@click.group(options_metavar='',
             subcommand_metavar='<command>',
             context_settings=settings)
def cli():
    """
    SOC: Data management system.
    """

# cli.add_command(MNIST)
