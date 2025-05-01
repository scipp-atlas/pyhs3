from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


class AppFilter(logging.Filter):
    def filter(self, record):
        record.filenameStem = Path(record.filename).stem
        return True


def rich_handler_factory():
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
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False,
        },
        "pyhs3": {
            "handlers": ["rich"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

__all__ = ()
