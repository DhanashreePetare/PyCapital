"""Exceptions and warnings raised by the public PyEquity API.

Error policy (one rule for every ``pyequity.api`` module):

- Connectors (``pyequity.connectors``) never raise on normal failures; they
  log and return an empty frame.
- The public API is stricter, because a silent empty result can be mistaken
  for "no data":
  * bad arguments raise ``InvalidArgumentError``;
  * if no source could answer at all, ``DataUnavailableError`` is raised;
  * data that is incomplete, from a fallback source, or empty is returned
    together with a warning that says exactly what is wrong.
"""


class PyEquityError(Exception):
    """Base class for exceptions raised by the public API."""


class InvalidArgumentError(PyEquityError, ValueError):
    """An argument is malformed (bad date, bad code, reversed range, ...)."""


class DataUnavailableError(PyEquityError):
    """No source could answer, so the library cannot say whether data exists."""


class PyEquityWarning(UserWarning):
    """Base class for warnings issued by the public API."""


class FallbackWarning(PyEquityWarning):
    """The data came from a fallback source, not the preferred one."""


class IncompleteDataWarning(PyEquityWarning):
    """The result is known to be missing rows (capped or failed requests)."""


class EmptyResultWarning(PyEquityWarning):
    """The source answered but returned no rows."""


class DataQualityWarning(PyEquityWarning):
    """The returned data failed one or more integrity checks."""