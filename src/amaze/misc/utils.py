""" Various utility functions from handling the Qt Application
"""

import os

from PyQt5.QtWidgets import QApplication


def qt_application(allow_create=True, start_offscreen=False):
    """Returns the currently running Qt application or creates a new one.

    :raises: RunTimeError if allow_create is False and no application exists.
    """
    if (app := QApplication.instance()) is None:
        if allow_create:
            args = []
            if start_offscreen:
                args.extend(["-platform", "offscreen"])
            app = QApplication(args)
        else:
            raise RuntimeError(
                "No QTApplication found. Create one first (in a large enough scope)"
            )
    return app


class NoQtApplicationException(EnvironmentError):
    def __init__(self):
        super().__init__(
            f"No Qt application created. Please use "
            f"{qt_application.__module__}.{qt_application.__qualname__}()"
            f" before manipulating any widgets."
        )


def has_qt_application():
    """Sanity check to ensure that a QtApplication exists"""
    if QApplication.instance() is None:
        raise NoQtApplicationException()
    return True


def qt_offscreen(offscreen=True):
    """Whether to request offscreen rendering from Qt (for headless environments)"""
    if offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    else:
        os.environ.pop("QT_QPA_PLATFORM", None)
