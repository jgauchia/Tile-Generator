"""Geofabrik region index.

The index published by Geofabrik describes every download it offers, with its
polygon.  Given the rectangle the user drew, this module decides which PBFs are
needed, how much they weigh and whether they fit on disk.
"""

import concurrent.futures
import json
import shutil
import time
import urllib.request
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from shapely.geometry import box, shape

INDEX_URL = "https://download.geofabrik.de/index-v1.json"
INDEX_FILE = "geofabrik-index-v1.json"
INDEX_MAX_AGE = 24 * 60 * 60
USER_AGENT = "tile-generator-web/1.0"
DOWNLOAD_TIMEOUT = 60
# The size probe must not hold the preview hostage: it is one byte, and the
# download servers answer it in half a second when they are healthy.
HEAD_TIMEOUT = 15

# Geofabrik rebuilds its extracts every day, so a cached PBF is only looked at
# again once it is this old: that bounds how stale the served data can be at one
# week, and it keeps the preview from asking anything about a fresh copy.
PBF_MAX_AGE = 7 * 24 * 60 * 60

# The size probes are one byte each and take half a second apiece, so a
# rectangle over a dozen regions is asked about all at once instead of waiting
# for each answer in turn.
PROBE_WORKERS = 8

# Measured against Geofabrik: 5 MB of cataluna in 1.001 s.  The plan uses it to
# say how long a download will take before it starts.
DOWNLOAD_BYTES_PER_SECOND = 5.0e6

# A rectangle covering more than this many regions is not a map but a continent:
# it is refused instead of sized and downloaded.
MAX_REGIONS = 12

# Measured amplification: the 268 MB Cataluna extract generates about 2.5 GB of
# packs.  The merge and the extract add one more copy of the input each.
OUTPUT_FACTOR = 10
INPUT_COPIES = 2
SAFETY_FACTOR = 1.25

# A region is only worth downloading when it covers at least this fraction of
# the requested rectangle.  Smaller leftovers are gaps between the published
# polygons, not areas worth pulling a whole country for.
MIN_COVER_FRACTION = 0.005


def bbox_polygon(bbox):
    """Shapely polygon for a (west, south, east, north) tuple."""
    return box(bbox[0], bbox[1], bbox[2], bbox[3])


def cache_name(url):
    """File name a Geofabrik URL keeps in the local cache."""
    return url.rstrip("/").rsplit("/", 1)[-1]


def _request(url, method):
    return urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})


def load_index(cache_dir, log=None, force=False):
    """Return the Geofabrik index, refreshing the cached copy once a day.

    A failed refresh keeps the cached copy: the service keeps working offline.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / INDEX_FILE
    age = time.time() - path.stat().st_mtime if path.is_file() else None
    if force or age is None or age > INDEX_MAX_AGE:
        try:
            if log:
                log("refreshing the Geofabrik index")
            download(INDEX_URL, path)
        except OSError as error:
            if not path.is_file():
                raise
            if log:
                log(f"keeping the cached index ({error})")
    return json.loads(path.read_text())


def pbf_regions(index):
    """Yield (id, name, parent, url, geometry) for every region with a PBF."""
    for feature in index["features"]:
        props = feature["properties"]
        url = props.get("urls", {}).get("pbf")
        geometry = feature.get("geometry")
        if not url or not geometry:
            continue
        yield props.get("id"), props.get("name"), props.get("parent"), url, shape(geometry)


def _ancestors(index_by_id, region_id):
    """Yield the ids of the parents of a region, nearest first."""
    parent = index_by_id.get(region_id, {}).get("parent")
    while parent:
        yield parent
        parent = index_by_id.get(parent, {}).get("parent")


def map_regions(index):
    """Yield (id, name, geometry) of the regions worth drawing on the map.

    Parents are left out: their children already tile them, and drawing both
    buries the map under stacked outlines.  Anything crossing the antimeridian
    goes as well, because Leaflet would draw it as a line across the whole map.
    """
    parents = {feature["properties"].get("parent") for feature in index["features"]}
    for feature in index["features"]:
        props = feature["properties"]
        geometry = feature.get("geometry")
        if not props.get("urls", {}).get("pbf") or not geometry:
            continue
        if props.get("id") in parents:
            continue
        shaped = shape(geometry)
        min_lon, _, max_lon, _ = shaped.bounds
        if max_lon - min_lon > 180:
            continue
        yield props.get("id"), props.get("name"), shaped


def select_regions(index, bbox):
    """Smallest set of regions whose polygons cover the rectangle.

    Regions are taken from the smallest polygon up, so a rectangle inside
    Cataluna only pulls Cataluna and one straddling France and Spain pulls the
    two border regions instead of the whole countries.

    A parent of an already chosen region is never added: the parent would cover
    whatever the region left, which is usually sea, at the price of a whole
    continent.  The leftover stays uncovered and is reported as coverage.
    """
    target = bbox_polygon(bbox)
    target_area = target.area
    index_by_id = {feature["properties"].get("id"): feature["properties"]
                   for feature in index["features"]}
    candidates = []
    for region_id, name, parent, url, geometry in pbf_regions(index):
        if not geometry.intersects(target):
            continue
        candidates.append(
            {
                "id": region_id,
                "name": name,
                "parent": parent,
                "url": url,
                "file": cache_name(url),
                "geometry": geometry,
                "area": geometry.area,
            }
        )
    candidates.sort(key=lambda candidate: candidate["area"])

    selected = []
    chosen = []
    remaining = target
    for candidate in candidates:
        if remaining.is_empty:
            break
        if any(candidate["id"] in _ancestors(index_by_id, region) for region in chosen):
            continue
        covered = candidate["geometry"].intersection(remaining).area
        if covered < target_area * MIN_COVER_FRACTION:
            continue
        selected.append(candidate)
        chosen.append(candidate["id"])
        remaining = remaining.difference(candidate["geometry"])
    coverage = 1.0 - remaining.area / target_area
    for candidate in selected:
        candidate.pop("geometry")
        candidate.pop("area")
    return selected, coverage


def _published_epoch(value):
    """Seconds since the epoch for an HTTP date, or None when it is not one."""
    if not value:
        return None
    try:
        stamp = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def remote_info(url):
    """Size and publication date of a Geofabrik file, from its first byte.

    A HEAD request is not usable any more: the download servers answer the
    redirect of a ``-latest.osm.pbf`` URL with the headers of the redirect
    itself and then never send those of the final response, so the probe sits
    there until the timeout and the plan fails.  Asking for a single byte
    follows the redirect, answers in half a second and reports the full size in
    Content-Range and the publication date in Last-Modified, both of them in
    the same request.
    """
    request = _request(url, "GET")
    request.add_header("Range", "bytes=0-0")
    with urllib.request.urlopen(request, timeout=HEAD_TIMEOUT) as response:
        published = _published_epoch(response.headers.get("Last-Modified"))
        content_range = response.headers.get("Content-Range")
        if content_range:
            size = int(content_range.rsplit("/", 1)[-1])
        else:
            size = int(response.headers.get("Content-Length") or 0)
    return size, published


def download(url, destination, progress=None, cancelled=None):
    """Download a file to its final name, continuing a partial one.

    A download that was cut leaves its ``.part`` file behind, and the next run
    asks for the rest of it with a Range header, so a region of hundreds of MB
    is not pulled from the beginning again.  The servers answer the range with
    a 206 and the bytes that are left; a 200 means the range was ignored and
    the file starts over.  The final file only appears when it is complete.
    """
    destination = Path(destination)
    part = destination.with_name(destination.name + ".part")
    done = part.stat().st_size if part.is_file() else 0
    request = _request(url, "GET")
    if done:
        request.add_header("Range", f"bytes={done}-")
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        if response.status == 206:
            mode = "ab"
        else:
            done = 0
            mode = "wb"
        total = done + int(response.headers.get("Content-Length") or 0)
        with part.open(mode) as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
                if cancelled is not None and cancelled.is_set():
                    raise OSError("cancelled by the user")
    part.replace(destination)


def cache_stem(filename):
    """Readable name of a cache file: andorra-latest.osm.pbf -> andorra."""
    stem = filename[:-len(".part")] if filename.endswith(".part") else filename
    if stem.endswith(".osm.pbf"):
        stem = stem[:-len(".osm.pbf")]
    if stem.endswith("-latest"):
        stem = stem[:-len("-latest")]
    return stem


def cache_deletions(cache_dir, keep):
    """Cached downloads that do not belong to the chosen area.

    Only PBFs and the partial files of an interrupted download are listed.  The
    water archive and the Geofabrik index live in the same directory and are
    never touched.  ``keep`` holds the file names the chosen area needs.
    """
    deletions = []
    for path in sorted(cache_dir.iterdir()):
        if path.name in keep or not path.is_file():
            continue
        partial = path.name.endswith(".part")
        if partial and path.name[: -len(".part")] in keep:
            # A cut download of a region this area needs: the run below
            # continues it instead of starting the file over.
            continue
        if not partial and not path.name.endswith(".pbf"):
            continue
        deletions.append(
            {
                "file": path.name,
                "name": cache_stem(path.name),
                "size": path.stat().st_size,
                "partial": partial,
            }
        )
    return deletions


def free_space(path):
    """Free bytes on the filesystem that holds the given path."""
    probe = Path(path).resolve()
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def required_space(total_input, missing_download):
    """Bytes the whole run needs free: the pending download plus the work copies."""
    transient = total_input * (INPUT_COPIES + OUTPUT_FACTOR)
    return int(missing_download + transient * SAFETY_FACTOR)


def plan(bbox, cache_dir, disk_path, extra_bytes=0):
    """Describe what generating the rectangle would download and how it fits.

    ``extra_bytes`` accounts for optional downloads the caller will make, such
    as the water polygons the first time they are needed.
    """
    cache_dir = Path(cache_dir)
    index = load_index(cache_dir)
    selected, coverage = select_regions(index, bbox)
    if len(selected) > MAX_REGIONS:
        raise ValueError(
            f"the area covers {len(selected)} regions, draw a smaller one")
    now = time.time()
    probes = []
    for region in selected:
        local = cache_dir / region["file"]
        if local.is_file():
            stat = local.stat()
            age = now - stat.st_mtime
            region.update(
                {
                    "size": stat.st_size,
                    "cached": True,
                    "age_days": round(age / 86400, 1),
                    "stale": False,
                    "probe_failed": False,
                }
            )
            if age > PBF_MAX_AGE:
                # Only a copy that already passed the threshold is worth a
                # request: a younger one cannot be a week behind whatever
                # Geofabrik has published since.
                probes.append((region, stat.st_mtime))
        else:
            region.update(
                {
                    "size": 0,
                    "cached": False,
                    "age_days": None,
                    "stale": False,
                    "probe_failed": False,
                }
            )
            probes.append((region, None))

    if probes:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(PROBE_WORKERS, len(probes))) as pool:
            running = {pool.submit(remote_info, region["url"]): (region, mtime)
                       for region, mtime in probes}
            for future in concurrent.futures.as_completed(running):
                region, mtime = running[future]
                try:
                    size, published = future.result()
                except OSError:
                    if mtime is None:
                        # A region that is not on disk needs its size to be
                        # sized and downloaded at all.
                        raise
                    region["probe_failed"] = True
                    continue
                if mtime is None:
                    region["size"] = size
                else:
                    region["stale"] = published is not None and published > mtime
                    if region["stale"]:
                        # The published file replaces the stale copy.
                        region["size"] = size

    regions = selected
    total = sum(region["size"] for region in regions)
    missing = sum(region["size"] for region in regions
                  if not region["cached"] or region["stale"])
    # A rectangle no region covers deletes nothing: there is no area to keep.
    deletions = cache_deletions(cache_dir, {region["file"] for region in regions}) if regions else []
    required = required_space(total, missing) + extra_bytes
    free = free_space(disk_path)
    return {
        "bbox": list(bbox),
        "coverage": coverage,
        "regions": regions,
        "deletes": deletions,
        "frees_bytes": sum(item["size"] for item in deletions),
        "total_bytes": total,
        "download_bytes": missing,
        "download_seconds": int(missing / DOWNLOAD_BYTES_PER_SECOND),
        "required_bytes": required,
        "free_bytes": free,
        "fits": free >= required,
    }


def _main():
    import sys

    bbox = [float(value) for value in sys.argv[1:5]]
    cache_dir = Path(__file__).resolve().parent.parent / "cache"
    report = plan(bbox, cache_dir, cache_dir.parent)
    print(f"coverage: {report['coverage'] * 100:.1f}%")
    for region in report["regions"]:
        mark = "cached" if region["cached"] else "download"
        print(f"  {mark:8} {region['id']:40} {region['size'] / 1e6:10.1f} MB")
    print(f"download: {report['download_bytes'] / 1e6:.1f} MB")
    print(f"required: {report['required_bytes'] / 1e6:.1f} MB")
    print(f"free:     {report['free_bytes'] / 1e6:.1f} MB")
    print(f"fits:     {report['fits']}")


if __name__ == "__main__":
    _main()
