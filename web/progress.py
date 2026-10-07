"""Live progress of a job, in the shape the page draws it.

Only one step of the chain says how far along it is: the tile generator
redraws a bar with the zoom, the percentage and the tiles it has packed, five
times a second (``src/tile_processor.hpp``).  The rest take minutes without a
number of their own, so the page shows the step that is running and, when that
step reports one, how much of it is done.

The tiles of a zoom are only known once that zoom has started, so there is no
total of tiles for the job as a whole: the engine groups the features into tiles
before packing them (``src/tile_processor.hpp``) and never prints a count for the
zooms that are still to come.  The page counts zooms for the whole job for that
reason, and this module keeps what the engine said of the zoom at hand.
"""

import re
import threading
import time

# One redraw of the bar of nav_generator: the zoom being built, how much of it,
# the tiles packed and the rate.  A chunk can carry several redraws of the same
# bar, and only the last one is current.
BAR = re.compile(r"Zoom\s+(\d+): \[[#-]*\]\s+(\d+)% \| *(\d+)/(\d+) tiles \| *([\d.]+) t/s")

# The line that closes a zoom, with how many tiles it took: the bar of the zoom
# is left at its real end, not at the last redraw it printed.
SUMMARY = re.compile(r"Zoom\s+(\d+):\s+(\d+) tiles \(")

# The steps of the chain, under the name the page shows for them.
STEPS = {
    "download": "Downloading the map data",
    "merge": "Merging the regions",
    "extract": "Clipping to the area",
    "water": "Preparing the water",
    "nav": "Building map",
    "route": "Building the routes",
}

# The same steps, under the name the box of statistics says them: the tiles are
# counted in tiles and the clip is the extract of osmium.
STEP_NAMES = {
    "plan": "plan",
    "download": "download",
    "merge": "merge",
    "extract": "clip",
    "water": "water",
    "nav": "tiles",
    "route": "routes",
}


class Progress:
    """What a job is doing now, and how far along that is."""

    def __init__(self, zoom_min, zoom_max):
        self._lock = threading.Lock()
        # The clock of the step that runs and the closed ones, so the job can say
        # afterwards where its time went; and the tiles the engine counted, which
        # it prints when it closes each zoom.
        self._steps = []
        self._step = None
        self._opened = None
        self._tiles = 0
        self._state = {
            "phase": "idle",
            "label": "",
            "percent": None,
            "done": None,
            "total": None,
            "rate": None,
            "zoom": None,
            "zoom_min": zoom_min,
            "zoom_max": zoom_max,
        }

    def step(self, phase, label=None):
        """Start a step: its numbers are unknown until it reports them."""
        with self._lock:
            self._close()
            self._step = phase
            self._opened = time.monotonic()
            self._state.update(
                phase=phase,
                label=label or STEPS.get(phase, phase),
                percent=None,
                done=None,
                total=None,
                rate=None,
                zoom=None,
            )

    def fraction(self, done, total):
        """A download reports the bytes it has and the bytes it needs."""
        with self._lock:
            self._state.update(
                percent=None if not total else min(100, round(done * 100 / total)),
                done=done,
                total=total,
            )

    def bar(self, chunk):
        """Read a chunk of an engine's output: its bar and its summaries."""
        with self._lock:
            for zoom, tiles in SUMMARY.findall(chunk):
                # The zoom is over: its bar ends at the tiles the engine counted,
                # not at the last redraw it printed.
                self._tiles += int(tiles)
                self._state.update(zoom=int(zoom), percent=100, done=int(tiles), total=int(tiles))
            matches = BAR.findall(chunk)
            if not matches:
                return
            zoom, percent, done, total, rate = matches[-1]
            self._state.update(
                zoom=int(zoom),
                percent=int(percent),
                done=int(done),
                total=int(total),
                rate=float(rate),
            )

    def close(self):
        """The chain is over: the step that was running stops counting."""
        with self._lock:
            self._close()

    def _close(self):
        """A step that ends adds its seconds; it is called with the lock held."""
        if self._step is None:
            return
        self._steps.append((self._step, time.monotonic() - self._opened))
        self._step = None

    def stats(self):
        """What the job did, for the box of figures the page shows once it is done.

        The seconds of a phase that ran more than once (a download per region)
        are added up, and the tiles are the ones the engine counted when it closed
        each zoom: both are measured, never estimated.
        """
        with self._lock:
            seconds = {}
            for phase, elapsed in self._steps:
                name = STEP_NAMES.get(phase, phase)
                seconds[name] = seconds.get(name, 0) + elapsed
            return {
                "tiles": self._tiles,
                "steps": [{"name": name, "seconds": round(elapsed, 1)}
                          for name, elapsed in seconds.items()],
            }

    def snapshot(self):
        """The state of the job, as the API hands it to the page."""
        with self._lock:
            return dict(self._state)
