from PyQt5.QtWidgets import QApplication


def application():
    """ Returns the currently running Qt application or creates a new one. """
    if (app := QApplication.instance()) is None:
        app = QApplication([])
    return app


class NoQtApplicationException(EnvironmentError):
    def __init__(self):
        super().__init__(
            f"No Qt application created. Please use "
            f"{application.__module__}.{application.__qualname__}()"
            f" before manipulating any widgets."
        )


def has_qt_application():
    """ Sanity check to ensure that a QtApplication exists """
    if QApplication.instance() is None:
        raise NoQtApplicationException()
    return True
