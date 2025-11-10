"""Various utility functions from handling the Qt Application"""

import os

from PyQt5.QtWidgets import QApplication


# pragma: exclude file

QT_PLATFORM_PLUGIN_KEY = "QT_QPA_PLATFORM"
QT_PLATFORM_OFFSCREEN_PLUGIN = "offscreen"


def qt_application(allow_create=True, start_offscreen=False):
    """Returns the currently running Qt application or creates a new one.

    :raises: RuntimeError if allow_create is False and no application exists.
    """
    if (app := QApplication.instance()) is None:
        if allow_create:
            if start_offscreen:
                qt_offscreen(offscreen=True)
            app = QApplication([])
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
    """Sanity check to ensure that a QtApplication exists

    :raises: NoQtApplicationException if no Qt application exists
    """
    if QApplication.instance() is None:
        raise NoQtApplicationException()
    return True


def qt_offscreen(offscreen=True):
    """Whether to request offscreen rendering from Qt (for headless environments)"""
    if offscreen:
        os.environ[QT_PLATFORM_PLUGIN_KEY] = QT_PLATFORM_OFFSCREEN_PLUGIN
    else:
        os.environ.pop(QT_PLATFORM_PLUGIN_KEY, None)


def is_qt_offscreen():
    """Tests whether the offscreen has been requested either through the environmental variable
    or through the Qt application itself"""
    if (platform := os.environ.get(QT_PLATFORM_PLUGIN_KEY)) is not None:
        return platform == QT_PLATFORM_OFFSCREEN_PLUGIN

    try:
        app = qt_application(allow_create=False)
        return app.platformName() == QT_PLATFORM_OFFSCREEN_PLUGIN
    except Exception:
        return False
