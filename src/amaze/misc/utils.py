from PyQt5.QtWidgets import QApplication


def qt_application(allow_create=True):
    """ Returns the currently running Qt application or creates a new one.

    :raises: RunTimeError if allow_create is False and not application exists.
    """
    if (app := QApplication.instance()) is None:
        if allow_create:
            app = QApplication([])
        else:
            raise RuntimeError("No QTApplication found. Create one first (in a"
                               " large enough scope)")
    return app


class NoQtApplicationException(EnvironmentError):
    def __init__(self):
        super().__init__(
            f"No Qt application created. Please use "
            f"{qt_application.__module__}.{qt_application.__qualname__}()"
            f" before manipulating any widgets."
        )


def has_qt_application():
    """ Sanity check to ensure that a QtApplication exists """
    if QApplication.instance() is None:
        raise NoQtApplicationException()
    return True
