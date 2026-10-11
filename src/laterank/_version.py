"""The installed laterank version.

A leaf module, so lower layers can read the version without importing `laterank/__init__.py`
(which will import the api layer).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("laterank")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"
