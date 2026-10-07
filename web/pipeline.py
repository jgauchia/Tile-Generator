"""Generation chain driven by the web service.

Every step shells out to a tool that already exists (osmium-tool, ogr2ogr and
the two generators); the engine itself is never modified.
"""

import io
import os
import shutil
import signal
import subprocess
import threading
import zipfile
from pathlib import Path

import geofabrik
from progress import Progress

# The ocean polygons published by osmdata.openstreetmap.de.  The archive is
# downloaded once and its contents are the shapefile the generator reads.
WATER_URL = "https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip"
WATER_ARCHIVE = "water-polygons-split-4326.zip"
WATER_ARCHIVE_BYTES = 906079605
WATER_EXTRACTED_BYTES = 1261046119


class PipelineError(RuntimeError):
    """A step of the chain could not be completed."""


class Pipeline:
    def __init__(self, repo_root, cache_dir, log, cancelled=None, progress=None):
        self.repo_root = Path(repo_root)
        self.cache_dir = Path(cache_dir)
        self.log = log
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.progress = progress if progress is not None else Progress(0, 0)
        self._progress_step = -1
        self._process = None

    def stop(self):
        """Kill the command that is running, if any.

        Each command opens its own session, so the signal reaches the process
        and whatever it may have started, and never the web service itself.
        """
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    def run(self, command, quiet=False):
        """Run a command in the repository root, streaming its output to the log.

        Every chunk reaches the log with the line ending it came with: the
        generators redraw their progress bar with a carriage return, and only
        the log can tell that redraw from a line of its own.  Universal
        newlines would turn it into a new line before it gets there.  The same
        chunk carries the numbers of that bar, which the page draws as a bar.

        ``quiet`` keeps the command line out of the log.  It is for the tools
        whose call is a wall of absolute paths and whose step the bar of the
        job already names, so the box of logs has nothing to gain from it; a
        command that fails still says which one it was.
        """
        if self.cancelled.is_set():
            raise PipelineError("cancelled by the user")
        command = [str(part) for part in command]
        if not quiet:
            self.log("$ " + " ".join(command))
        process = subprocess.Popen(
            command,
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._process = process
        try:
            output = io.TextIOWrapper(process.stdout, encoding="utf-8",
                                      errors="replace", newline="")
            for chunk in output:
                self.log(chunk)
                self.progress.bar(chunk)
            code = process.wait()
        finally:
            self._process = None
        if self.cancelled.is_set():
            raise PipelineError("cancelled by the user")
        if code != 0:
            raise PipelineError(f"{Path(command[0]).name} exited with code {code}")

    def binary(self, name):
        """Locate a generator: the local build first, then PATH."""
        local = self.repo_root / "build" / name
        if local.is_file():
            return local
        found = shutil.which(name)
        if not found:
            raise PipelineError(f"{name} not found: build it or add it to PATH")
        return found

    def _progress(self, done, total):
        self.progress.fraction(done, total)
        if not total:
            return
        step = done * 10 // total
        if step != self._progress_step:
            self._progress_step = step
            self.log(f"  {step * 10}%")

    def fetch_regions(self, regions):
        """Delete what the area does not need and download what it is missing.

        The deletions are the ones the plan announced before the button was
        pressed.  A cached copy that the published file has left behind is
        downloaded again, over itself, so a failed download still leaves the
        old copy in place.  A copy the freshness probe could not confirm is
        kept: a check that failed is not a reason to throw data away.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.progress.step("download", "Checking the cached copies")
        keep = {region["file"] for region in regions}
        for item in geofabrik.cache_deletions(self.cache_dir, keep):
            reason = "partial download" if item["partial"] else "another area"
            self.log(f"delete: {item['name']} ({item['size'] / 1e6:.1f} MB, {reason})")
            (self.cache_dir / item["file"]).unlink()
        files = []
        for region in regions:
            path = self.cache_dir / region["file"]
            if path.is_file() and region["stale"]:
                self.log(f"refresh: {region['name']} ({region['size'] / 1e6:.1f} MB, "
                         f"replacing the {region['age_days']:.1f} days old copy)")
                stale = True
            else:
                stale = False
            if path.is_file() and not stale:
                age = "" if region["age_days"] is None else f", {region['age_days']:.1f} days old"
                self.log(f"cached: {region['file']} ({path.stat().st_size / 1e6:.1f} MB{age})")
            else:
                if not stale and not path.is_file():
                    part = path.with_name(path.name + ".part")
                    resumed = f", resuming {part.stat().st_size / 1e6:.1f} MB" if part.is_file() else ""
                    self.log(f"download: {region['name']} ({region['size'] / 1e6:.1f} MB, "
                             f"not cached{resumed})")
                verb = "Refreshing" if stale else "Downloading"
                self.progress.step("download", f"{verb} {region['name']}")
                self._progress_step = -1
                geofabrik.download(region["url"], path, self._progress, self.cancelled)
            files.append(path)
        return files

    def merge(self, files, destination):
        """Combine the inputs into one file with one version of every object.

        Geofabrik extracts share whole ways and relations along their borders,
        and a cached copy from another day carries another version of them.
        ``osmium merge`` keeps every version it is given (its manual documents
        it for history files) and ``osmium extract`` refuses an input with a
        repeated ID ("Way ID twice in input"), so the merge is followed by a
        pass of ``osmium time-filter`` with no instant: the newest version of
        each object and nothing else, which is the file the engine can read.
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if len(files) == 1:
            self.log(f"single region, no merge needed: {files[0].name}")
            return files[0]
        if destination.exists():
            destination.unlink()
        self.progress.step("merge", f"Merging {len(files)} regions")
        self.run(["osmium", "merge", *files, "-o", destination, "--no-progress"], quiet=True)
        unique = destination.with_name("unique.osm.pbf")
        if unique.exists():
            unique.unlink()
        self.progress.step("merge", "Dropping the repeated objects")
        self.run(["osmium", "time-filter", destination, "-o", unique, "--no-progress"], quiet=True)
        # The merged file has served its purpose; only the deduplicated one
        # goes on to the clip, and the disk guard counts two copies of the
        # input, not three.
        destination.unlink()
        return unique

    def extract(self, source, bbox, destination):
        """Clip to the rectangle with complete ways.

        The ``simple`` strategy is not an option: the engine aborts with
        "location for one or more nodes not found" as soon as a way references
        a node the clip dropped.  Keeping whole ways is what makes the clip safe
        for the engine, at the price of a small margin around the rectangle.
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        bbox_arg = ",".join(f"{value:.7f}" for value in bbox)
        self.progress.step("extract")
        self.run(["osmium", "extract", "--bbox", bbox_arg, "--strategy", "complete_ways",
                  "--no-progress", "-o", destination, source], quiet=True)
        return destination

    def ensure_water(self, shapefile):
        """Return the water shapefile, downloading and unpacking it once."""
        shapefile = Path(shapefile)
        if shapefile.is_file():
            return shapefile
        archive = self.cache_dir / WATER_ARCHIVE
        if not archive.is_file():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.log("download: water polygons (about 900 MB, needed once)")
            self.progress.step("water", "Downloading the water polygons")
            self._progress_step = -1
            geofabrik.download(WATER_URL, archive, self._progress, self.cancelled)
        self.log("extracting the water polygons")
        self.progress.step("water", "Unpacking the water polygons")
        if self.cancelled.is_set():
            raise PipelineError("cancelled by the user")
        with zipfile.ZipFile(archive) as package:
            package.extractall(self.repo_root)
        if not shapefile.is_file():
            raise PipelineError(f"the water archive did not contain {shapefile.name}")
        return shapefile

    def clip_water(self, shapefile, bbox, destination_dir):
        """Clip the ocean shapefile to the rectangle with ogr2ogr."""
        destination_dir = Path(destination_dir)
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        destination_dir.mkdir(parents=True)
        self.progress.step("water", "Clipping the water to the area")
        self.run(["ogr2ogr", "-clipsrc", *[f"{value:.7f}" for value in bbox],
                  "-f", "ESRI Shapefile", destination_dir, shapefile], quiet=True)
        candidates = sorted(destination_dir.glob("*.shp"))
        if not candidates:
            raise PipelineError("the water clip produced no shapefile")
        return candidates[0]

    def generate_nav(self, source, output_dir, features, zoom, water):
        self.progress.step("nav")
        command = [self.binary("nav_generator"), source, output_dir, features,
                   "--zoom", f"{zoom[0]}-{zoom[1]}"]
        if water:
            command += ["--water-shp", water]
        self.run(command)

    def generate_route(self, source, output_dir):
        self.progress.step("route")
        self.run([self.binary("route_generator"), source, output_dir])
