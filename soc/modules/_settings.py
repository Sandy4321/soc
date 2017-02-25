"""_settings.py

Defines methods for accessing user settings.
"""

from __future__ import absolute_import
from __future__ import print_function

import json
import logging
import os
import warnings

# Loads the base directory (Unix or Windows).
_base_dir = os.path.expanduser('~')
if not os.access(_base_dir, os.W_OK):
    _base_dir = '/tmp'

# Gets or creates the main SOC directory.
_pysoc_dir = os.path.join(_base_dir, '.pysoc')
if not os.path.exists(_pysoc_dir):
    os.makedirs(_pysoc_dir)

# Loads settings file.
_settings_path = os.path.join(_pysoc_dir, 'settings.json')
if os.path.exists(_settings_path):
    _settings_dict = json.load(open(_settings_path))

# Creates a new settings file.
else:
    _default_data_dir = os.path.join(_pysoc_dir, 'data')
    if not os.path.exists(_default_data_dir):
        os.makedirs(_default_data_dir)

    _settings_dict = {
        'data_dir': _default_data_dir,
        'chunk_size': 8192,
    }
    print('creating settings dict')
    print('settings dict:', _settings_dict)
    with open(_settings_path, 'w') as f:
        f.write(json.dumps(_settings_dict, indent=4))


# Overrides with environment variables.
for key, value in _settings_dict.items():
    var_name = 'PYSOC_%s' % key.upper()
    if var_name not in os.environ:
        continue

    var_value = os.environ[var_name]
    if isinstance(value, (list, tuple)):
        var_value = var_value.split(',')

    # Updates settings with the new value.
    _settings_dict[key] = var_value
    logging.info('Updated ["%s": "%s" => "%s"]', key, value, var_value)


def _check_settings_dict():
    """Performs validation checks on the settings dictionary."""

    if 'chunk_size' not in _settings_dict:
        raise ValueError('Settings should include "chunk_size", an integer '
                         'specifying the download chunk size. Set this value '
                         'in "%s"' % _settings_path)

    if not isinstance(_settings_dict['chunk_size'], int):
        raise ValueError('Expected chunk_size to be an integer, got "%s"'
                         % str(_settings_dict['chunk_size']))

    if 'data_dir' not in _settings_dict:
        raise ValueError('The specified settings dictionary does not specify '
                         'a data directory: "%s" This path can be specified '
                         'in "%s"' % (str(_settings_dict), _settings_path))

    if not os.access(_settings_dict['data_dir'], os.R_OK):
        raise ImportError('The specified data_dir does not have read '
                          'permission: "%s" This directory can be specified '
                          'in "%s"' % (_settings_dict['data_dir'], _settings_path))

    if not os.access(_settings_dict['data_dir'], os.W_OK):
        warnings.warn('The current data_dir does not have write '
                      'permission: "%s". You will therefore be unable to '
                      'download new files.' % _settings_dict['data_dir'])

_check_settings_dict()


def get_module_subdir(module_name):
    """Returns the path to a module's subdirectory."""

    subdir_path = os.path.join(_settings_dict['data_dir'], module_name)

    # Creates the data subdirectory if it doesn't exist.
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

    return subdir_path


def get_setting(attribute):
    """Gets the value of an attribute."""

    attribute = attribute.lower()

    if attribute not in _settings_dict:
        raise ValueError('Invalid attribute: "%s". Available attributes: '
                         '[%s]' % (attribute, ', '.join(_settings_dict.keys())))

    return _settings_dict[attribute]


def set_setting(key, value):
    """Updates a setting value."""

    key = key.lower()

    if key not in _settings_dict:
        raise ValueError('Cannot add "%s" to the settings dictionary. '
                         'Available properties: "%s"'
                         % (key, _settings_dict.items()))

    _settings_dict[key] = value

    # Perform checks on the updated value.
    _check_settings_dict()
