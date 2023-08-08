import sys
from io import TextIOBase
from pathlib import Path
from typing import Optional, Callable

from colorama import Fore, Style


class Tee:
    """Ensure that everything that's printed is also saved
    """

    class PassthroughStream(TextIOBase):
        """Forwards received messages to log/file"""
        def __init__(self, parent: 'Tee'):  self.tee = parent
        def write(self, msg):   self.tee.write(msg)
        def flush(self):    self.tee.flush()
        def isatty(self): return self.tee.out.isatty()
        def close(self): pass

    class FormattedStream(PassthroughStream):
        """Forwards received messages to log/file """
        def __init__(self, parent: 'Tee', formatter: str):
            super().__init__(parent)
            self.formatter = formatter

        def write(self, msg):
            super().write(self.formatter.format(msg))

    _Filterer = Optional[Callable[[str], bool]]

    def __init__(self,
                 path: Path,
                 filter_out: _Filterer = lambda _: False):
        self.out = sys.stdout
        self.log = open(path, 'w')
        sys.stdout = self.PassthroughStream(self)
        sys.stderr = self.FormattedStream(
            self, Fore.RED + "{}" + Style.RESET_ALL)
        self.filter = filter_out

        # print(f"Redirecting stdout output to {path}")
        # print(f"Redirecting stderr output to {path}", file=sys.stderr)
        assert path.exists()

    def __del__(self):
        self.flush()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def write(self, msg: str):
        if not self.filter(msg):
            self.log.write(msg)
            self.out.write(msg)

    def flush(self):
        self.out.flush()
        self.log.flush()

# class Tee:
#     """Ensure that everything that's printed is also saved
#     """
#
#     class PassthroughStream:
#         """Forwards received messages to log/file"""
#         def __init__(self, parent: 'Tee'):  self.tee = parent
#         def write(self, msg):   self.tee.write(msg)
#         def flush(self):    self.tee.flush()
#         def isatty(self): return self.tee.out.isatty()
#         def close(self): pass
#
#     class FormattedStream(PassthroughStream):
#         """Forwards received messages to log/file """
#         def __init__(self, parent: 'Tee', formatter: str):
#             super().__init__(parent)
#             self.formatter = formatter
#
#         def write(self, msg):
#             super().write(self.formatter.format(msg))
#
#     _Filterer = Optional[Callable[[str], bool]]
#
#     def __init__(self, filter_out: _Filterer = lambda _: False):
#         self.out = sys.stdout
#         self.log = None
#         self.msg_queue = []    # Collect until log file is available
#         self.registered = False
#         self.filter = filter_out
#
#     def register(self):
#         if not self.registered:
#             sys.stdout = self.PassthroughStream(self)
#             sys.stderr = self.FormattedStream(
#                 self, Fore.RED + "{}" + Style.RESET_ALL)
#             self.registered = True
#
#     def teardown(self):
#         if self.registered:
#             self.flush()
#             sys.stdout = sys.__stdout__
#             sys.stderr = sys.__stderr__
#             self.registered = False
#
#     def set_log_path(self, path: Path):
#         self.register()
#         self.log = open(path, 'w')
#         for msg in self.msg_queue:
#             self._write(msg)
#
#     def _write(self, msg: str):
#         if not self.filter(msg):
#             self.log.write(msg)
#
#     def write(self, msg: str):
#         if self.log is None:
#             self.msg_queue.append(msg)
#         else:
#             self._write(msg)
#         self.out.write(msg)
#
#     def flush(self):
#         self.out.flush()
#         if self.log is not None:
#             self.log.flush()
