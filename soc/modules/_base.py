"""_base.py

Defines the base template for a module. To add a new module, you should
override the Module class or one of its subclasses.
"""

from __future__ import absolute_import

from ._settings import get_setting, get_module_subdir

import click
import logging
import os
import uuid
from urllib2 import urlopen, Request, HTTPError


class Module(object):
    """Defines the abstract module class.

    TODO: Documentation.
    """

    def __init__(self, batch_size=32):
        self.module_name = self.__class__.__name__.lower()
        self.data_subdir = get_module_subdir(self.module_name)
        self.batch_size = batch_size

    def get_path(self, fname):
        """Returns the path to the specified module file.

        Args:
            fname: str, the name of the requested module file.

        Returns:
            fpath: str, the full path to the file.
        """

        fpath = os.path.join(self.data_subdir, fname)
        return fpath

    def get_unique(self, ext):
        """Returns a unique filename with the desired extension.

        Args:
            ext: str, the extension to use.

        Returns:
            a unique file path in this module with the right extension.
        """

        return self.get_path('%s.%s' % (str(uuid.uuid4()), ext))

    def validate_file(self, fname, md5_hash):
        """Validates a file's MD5 hash, as done in Keras.

        Args:
            fpath: str, path to the file to validate.
            md5_hash: the md5 hash being validated against.

        Returns:
            is_valid: bool, True if the file is valid.
        """

        fpath = self.get_file_path(fname)

        hasher = hashlib.md5()
        with open(fpath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        is_valid = str(hasher.hexdigest()) == str(md5_hash)

        return is_valid


    def get_file(self, fname, url, use_bar=True, download=False):
        """Retrieves a file from the specified URL, or loads it if it exists.

        Args:
            fname: str, name of the file to download.
            url: str, original URL of the file.
            use_bar: bool, whether or not to use the progress bar.
            download: bool, if set, always download a new file.

        Returns:
            fpath: str, path to the downloaded file.
        """

        fpath = self.get_file_path(fname)

        # Automatically download if no cache exists, or the MD5 hash is bad.
        if not os.path.exists(fpath):
            download = True
        else:
            if md5_hash is not None:
                if not self._validate_file(fpath, md5_hash):
                    print('A local file was found, but it seems to be '
                          'incomplete or outdated.')
                    download = True

        response = urlopen(url)
        info, url = response.info(), response.url

        # Gets the total file size from the header.
        fsize = int(info.get('Content-Length').strip())

        if use_bar:
            bar = click.progressbar(length=fsize, label=self.module_name)

        # Downloads the file.
        chunk_size = get_setting('chunk_size')
        with open(fpath, 'wb') as f:
            chunk = response.read(chunk_size)
            while chunk is not None:
                f.write(chunk)
                if use_bar:
                    bar.update(chunk_size)
                chunk = response.read(chunk_size)

        if use_bar:
            bar.close()

        return fpath

    def download(self):
        """Downloads the data and performs all preprocessing.

        Child methods should override this method. The command-line tool calls
        this method to download things.
        """
        raise NotImplementedError()

    def pd(self):
        raise NotImplementedError()
