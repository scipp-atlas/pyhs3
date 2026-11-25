"""
Logging for pyhs3.

pyhs3 is a library, so importing it configures no logging at all: the root
logger, every other library's logger, and the ``pyhs3`` logger itself are left
as the application configured them.  The ``pyhs3`` logger deliberately gets no
handler, not even a :class:`logging.NullHandler`, so that without any
configuration Python's last-resort handler still prints pyhs3's warnings and
errors to ``stderr``.  :func:`setup` gives the ``pyhs3`` logger a rich console
handler instead; the ``pyhs3`` command-line tool calls it.
"""

from __future__ import annotations

import logging
from logging import LogRecord
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


class AppFilter(logging.Filter):
    """
    AppFilter
    """

    def filter(self, record: LogRecord) -> bool:
        record.filenameStem = Path(record.filename).stem
        return True


def rich_handler_factory() -> RichHandler:
    return RichHandler(
        console=Console(width=160, stderr=True),
        rich_tracebacks=True,
        tracebacks_suppress=[],
        markup=True,
    )


def setup(level: int | str = logging.INFO) -> None:
    """
    Send pyhs3's log records to a rich console handler.

    Configures only the ``pyhs3`` logger: it sets its level and gives it one
    :class:`~rich.logging.RichHandler`.  The root logger and other loggers are
    left untouched, and records still propagate, so handlers the application
    attaches to the root logger (or pytest's ``caplog``) see them too.
    Calling it again updates the level without adding a second handler.

    Args:
        level: Level of the ``pyhs3`` logger.
    """
    logger = logging.getLogger("pyhs3")
    logger.setLevel(level)
    if not any(isinstance(handler, RichHandler) for handler in logger.handlers):
        handler = rich_handler_factory()
        handler.setFormatter(
            logging.Formatter("[[yellow]%(filenameStem)s[/]] %(message)s")
        )
        handler.addFilter(AppFilter())
        logger.addHandler(handler)


__all__ = ("setup",)
