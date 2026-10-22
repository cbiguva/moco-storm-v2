"""Logging setup shared by the command-line entry points."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_file: Optional[Path] = None, verbose: bool = True) -> None:
    """Log to stderr and, if given, to ``log_file``. Safe to call more than once."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.INFO if verbose else logging.WARNING)

    formatter = logging.Formatter(_FORMAT, _DATEFMT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Third-party loggers that are noisy at INFO.
    for name in ("matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
