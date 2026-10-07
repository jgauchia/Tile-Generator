"""Stream a generated pack as a ZIP without duplicating it on disk."""

import io
import queue
import threading
import zipfile
from pathlib import Path


class _Stream(io.RawIOBase):
    """Write-only stream that hands chunks to the HTTP response as they appear."""

    def __init__(self):
        self.chunks = queue.Queue()
        self.offset = 0

    def writable(self):
        return True

    def seekable(self):
        return False

    def write(self, data):
        data = bytes(data)
        self.offset += len(data)
        self.chunks.put(data)
        return len(data)

    def tell(self):
        return self.offset

    def flush(self):
        pass

    def close(self):
        self.chunks.put(None)
        super().close()


def stream_zip(directory, arcname=""):
    """Yield a stored ZIP of the directory, built while the client downloads it."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    stream = _Stream()

    def build():
        try:
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
                for path in sorted(directory.rglob("*")):
                    if path.is_file():
                        relative = Path(arcname) / path.relative_to(directory)
                        archive.write(path, str(relative))
        finally:
            stream.close()

    threading.Thread(target=build, daemon=True).start()
    while True:
        chunk = stream.chunks.get()
        if chunk is None:
            break
        yield chunk


def _main():
    import sys

    with open(sys.argv[2], "wb") as handle:
        for chunk in stream_zip(sys.argv[1]):
            handle.write(chunk)


if __name__ == "__main__":
    _main()
