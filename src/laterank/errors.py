"""laterank's exceptions.

A leaf module: it imports only the standard library, so every layer may import it. Every message
says what happened, then what to do about it.
"""


class LaterankError(Exception):
    """Base class of every error laterank raises on purpose."""


class IndexFormatError(LaterankError):
    """An index folder can't be read: a missing or corrupt file, pickled data, or a manifest this
    release doesn't understand."""
