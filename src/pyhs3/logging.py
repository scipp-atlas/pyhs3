from __future__ import annotations

import logging
import logging.config
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
        console=Console(width=160),
        rich_tracebacks=True,
        tracebacks_suppress=[],
        markup=True,
    )


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "filters": {
        "appfilter": {
            "()": AppFilter,
        }
    },
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
        "pretty": {"format": "[[yellow]%(filenameStem)s[/]] %(message)s"},
    },
    "handlers": {
        "default": {
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",  # Default is stderr
        },
        "rich": {
            "()": rich_handler_factory,
            "formatter": "pretty",
            "filters": ["appfilter"],
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "rich"],
            "level": "WARNING",
            "propagate": False,
        },
        "pyhs3": {
            "handlers": [],
            "level": "INFO",
            "propagate": True,
        },
    },
}


def setup() -> None:
    """
    Initialize logging based on the configuration dictionary in this file.
    """
    logging.config.dictConfig(LOGGING_CONFIG)


__all__ = ("setup",)
