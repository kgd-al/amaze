""" Implementation of a python guard to prevent negative interactions
between opencv2 and PyQT5"""

import os

from PyQt5.QtCore import QLibraryInfo


class CV2QTGuard:
    """Acts as a guard allowing both PyQt5 and opencv-python to use the
     xcb.qpa plugin without confusion.

     Temporarily restores environmental variable "QT_QPA_PLATFORM_PLUGIN_PATH"
     to the value used by qt, taken from
     QLibraryInfo.location(QLibraryInfo.PluginsPath)
     """

    QPA_PATH_NAME = "QT_QPA_PLATFORM_PLUGIN_PATH"
    QPA_PLATFORM_NAME = "QT_QPA_PLATFORM"

    def __init__(self, platform=True, path=True):
        self._qta_platform, self._qta_path = platform, path
        self.qta_platform, self.qta_path = None, None

    @staticmethod
    def _save_and_replace(key, override):
        value = os.environ.get(key, None)
        os.environ[key] = override
        return value

    def __enter__(self):
        if self._qta_platform:
            self.qta_platform = self._save_and_replace(
                self.QPA_PLATFORM_NAME, "offscreen")
        if self._qta_path:
            self.qta_path = self._save_and_replace(
                self.QPA_PATH_NAME,
                QLibraryInfo.location(QLibraryInfo.PluginsPath))

    @staticmethod
    def _restore_or_clean(key, saved_value):
        if isinstance(saved_value, str):
            os.environ[key] = saved_value
        else:
            os.environ.pop(key)

    def __exit__(self, *_):
        if self._qta_platform:
            self._restore_or_clean(self.QPA_PLATFORM_NAME, self.qta_platform)
        if self._qta_path:
            self._restore_or_clean(self.QPA_PATH_NAME, self.qta_path)
        return False
