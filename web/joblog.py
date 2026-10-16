"""Job log that keeps a redrawn progress bar on a single line.

The generators draw their progress with a carriage return and an ANSI erase
sequence, several times a second: each refresh is the same line over again.
Written as it comes, a plain log file keeps one line per refresh and a single
zoom fills hundreds of them with the same bar.  This writer gives a chunk that
ends in a carriage return the meaning it has on a terminal: the line being
drawn is replaced by it, so the file keeps one line per bar and the reader
watches it advance.
"""

import os
import re
from pathlib import Path

# What the generators send before every redraw: ESC [ 2K clears the line.
ERASE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class JobLog:
    """Append finished lines to a file, rewriting the line that is still open."""

    def __init__(self, path):
        self.path = Path(path)
        self._handle = self.path.open("wb")
        # Text of the line being drawn and the bytes of it already on disk.
        self._open = None
        self._stored = 0

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def write(self, message):
        """Take one chunk of output, with the line ending it came with."""
        text = ERASE.sub("", message)
        ending = ""
        for candidate in ("\r\n", "\n", "\r"):
            if text.endswith(candidate):
                ending = candidate
                text = text[: -len(candidate)]
                break
        if ending == "\r":
            # The generator is redrawing the bar: the previous draw leaves.
            self._open = text
            self._store()
            return
        if ending:
            # The last draw of the bar, ended by the generator itself.
            self._open = text
            self._store()
            self._close()
            return
        # A message of the service: it never shares a line with a bar.
        self._close()
        self._open = text
        self._store()
        self._close()

    def close(self):
        self._close()
        self._handle.close()

    def _store(self):
        """Make the open line be the one on disk, whatever was there before."""
        if self._stored:
            self._handle.seek(-self._stored, os.SEEK_CUR)
            self._handle.truncate()
        data = self._open.encode("utf-8")
        self._handle.write(data)
        self._handle.flush()
        self._stored = len(data)

    def _close(self):
        if self._open is None:
            return
        self._handle.write(b"\n")
        self._handle.flush()
        self._open = None
        self._stored = 0
