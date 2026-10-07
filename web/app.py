#!/usr/bin/env python3
"""Web service that turns a rectangle on the map into a tile pack.

The service only orchestrates: Geofabrik downloads, osmium merge and extract,
and the two generators that already ship in the image.
"""

import json
import math
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

import archive
import geofabrik
from joblog import JobLog
from pipeline import (Pipeline, PipelineError, WATER_ARCHIVE_BYTES,
                      WATER_EXTRACTED_BYTES)
from progress import Progress

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
CACHE_DIR = REPO_ROOT / "cache"
# The job logs live outside the output directory: the packs of a job that has
# been downloaded are deleted, and its log should outlive them.
LOG_DIR = REPO_ROOT / "logs"
DEFAULT_FEATURES = "features.json"
DEFAULT_WATER_SHAPE = "water-polygons-split-4326/water_polygons.shp"
LOG_TAIL_LINES = 200
ZOOM_MIN = 0
ZOOM_MAX = 20
# The name of the route pack ends up in the name of the file the browser saves,
# so it is kept to what a file name carries: letters, digits, spaces, dots,
# dashes and underscores.
PLAIN_NAME = re.compile(r"^[A-Za-z0-9 ._-]{1,80}$")

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

_jobs = {}
_lock = threading.Lock()
_regions_json = None


def _repo_path(value):
    return (REPO_ROOT / value).resolve()


def _existing_file(label, value):
    if not value:
        raise ValueError(f"{label} is required")
    path = _repo_path(value)
    if not path.is_file():
        raise ValueError(f"{label} not found: {value}")
    return path


def _water_extra(payload):
    """Bytes the water polygons add when they still have to be fetched."""
    if not payload.get("water") or _repo_path(DEFAULT_WATER_SHAPE).is_file():
        return 0
    return WATER_ARCHIVE_BYTES + WATER_EXTRACTED_BYTES


def _log_lines(log_path):
    """The tail of the job log, as the page shows it."""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_TAIL_LINES:]
    except FileNotFoundError:
        return []


def _parse_bbox(payload):
    bbox = payload.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [west, south, east, north]")
    try:
        west, south, east, north = (float(value) for value in bbox)
    except (TypeError, ValueError):
        raise ValueError("bbox values must be numbers")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox is out of range or empty")
    return [west, south, east, north]


def _relative(path):
    path = Path(path)
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _human_seconds(seconds):
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min"
    return f"{seconds / 3600:.1f} h"


def _dir_bytes(directory):
    """What a folder of packs weighs, or None while it does not exist."""
    if not directory.is_dir():
        return None
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _area_km2(bbox):
    """The area of the rectangle, near enough: a degree of latitude is 110.57 km
    everywhere and a degree of longitude is 111.32 km shrunk by the cosine of the
    latitude (so a degree is 80.06 km at 44 degrees north).  Under one percent off
    for the rectangles this tool is given."""
    west, south, east, north = bbox
    middle = math.radians((south + north) / 2)
    return (east - west) * 111.32 * math.cos(middle) * (north - south) * 110.57


def _stats(job):
    """What a job that is over did: its time and where that time went, the tiles
    the engine counted, and what the packs it wrote weigh."""
    measured = job["progress"].stats()
    return {
        "seconds": round(job["finished"] - job["started"], 1) if job["finished"] else None,
        "steps": measured["steps"],
        "tiles": measured["tiles"],
        "navmap_bytes": _dir_bytes(job["navmap_dir"]),
        "route_bytes": _dir_bytes(job["route_dir"]) if job["route"] else None,
        "bbox": job["bbox"],
        "area_km2": round(_area_km2(job["bbox"]), 1) if job["bbox"] else None,
    }


def _describe(plan):
    download = f"download {plan['download_bytes'] / 1e6:.1f} MB of {plan['total_bytes'] / 1e6:.1f} MB"
    if plan["download_seconds"]:
        download += f" (about {_human_seconds(plan['download_seconds'])})"
    lines = [f"area {plan['bbox'][0]:.4f},{plan['bbox'][1]:.4f} -> {plan['bbox'][2]:.4f},{plan['bbox'][3]:.4f}",
             f"coverage {plan['coverage'] * 100:.1f}%",
             download,
             f"needs {plan['required_bytes'] / 1e9:.2f} GB free, {plan['free_bytes'] / 1e9:.2f} GB available"]
    for region in plan["regions"]:
        if region["stale"]:
            state = "refresh"
        elif region["cached"]:
            state = "cached"
        else:
            state = "download"
        age = "" if region["age_days"] is None else f", {region['age_days']:.1f} days old"
        if region["probe_failed"]:
            age += ", check failed"
        lines.append(f"  {state:8} {region['name']} ({region['size'] / 1e6:.1f} MB{age})")
    if plan["deletes"]:
        lines.append(f"delete:  {plan['frees_bytes'] / 1e6:.1f} MB of cache from another area")
        for item in plan["deletes"]:
            mark = "partial" if item["partial"] else "other area"
            lines.append(f"  {mark:11} {item['name']} ({item['size'] / 1e6:.1f} MB)")
    return lines


def _run_job(job, spec):
    with JobLog(job["log_path"]) as log:
        pipeline = Pipeline(REPO_ROOT, CACHE_DIR, log.write, job["cancel"], job["progress"])
        job["pipeline"] = pipeline
        try:
            if spec["bbox"] is not None:
                # Reading the index and probing the regions takes seconds of its
                # own: it is a step with a clock like any other.
                job["progress"].step("plan", "Reading the region index")
                plan = geofabrik.plan(spec["bbox"], CACHE_DIR, REPO_ROOT, _water_extra(spec))
                if not plan["regions"]:
                    raise PipelineError("no published region covers this area")
                for line in _describe(plan):
                    log.write(line)
                if not plan["fits"]:
                    raise PipelineError(
                        f"not enough free disk space: needs {plan['required_bytes'] / 1e9:.2f} GB, "
                        f"has {plan['free_bytes'] / 1e9:.2f} GB")
                files = pipeline.fetch_regions(plan["regions"])
                work = job["output_dir"] / "work"
                merged = pipeline.merge(files, work / "merged.osm.pbf")
                source = pipeline.extract(merged, spec["bbox"], work / "clip.osm.pbf")
            else:
                source = Path(spec["input"])
            log.write(f"input: {source}")

            water = None
            if spec["water"]:
                water = pipeline.ensure_water(REPO_ROOT / DEFAULT_WATER_SHAPE)
                if spec["bbox"] is not None:
                    water = pipeline.clip_water(water, spec["bbox"],
                                                job["output_dir"] / "work" / "water")

            pipeline.generate_nav(source, job["navmap_dir"], spec["features"], spec["zoom"], water)
            if spec["route"]:
                # route_generator appends ROUTE/<profile> to the directory it is given.
                pipeline.generate_route(source, job["output_dir"])

            log.write("done")
            job["state"] = "done"
        except (PipelineError, OSError, ValueError) as error:
            if job["cancel"].is_set():
                log.write("cancelled")
                job["error"] = "cancelled by the user"
                job["state"] = "cancelled"
            else:
                log.write(f"ERROR: {error}")
                job["error"] = str(error)
                job["state"] = "failed"
        finally:
            job["pipeline"] = None
            # The chain is over: the last step stops counting and the clock of the
            # job closes, which is what the box of statistics reads afterwards.
            job["progress"].close()
            job["finished"] = time.time()


@app.get("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.get("/api/config")
def config():
    return jsonify(
        {
            "features": DEFAULT_FEATURES,
            "water_ready": _repo_path(DEFAULT_WATER_SHAPE).is_file(),
            "zoom_min": 6,
            "zoom_max": 17,
            "free_bytes": geofabrik.free_space(REPO_ROOT),
        }
    )


@app.get("/api/regions")
def regions():
    global _regions_json
    if _regions_json is None:
        try:
            index = geofabrik.load_index(CACHE_DIR)
        except OSError as error:
            return jsonify({"error": f"cannot reach the Geofabrik index: {error}"}), 503
        features = []
        for region_id, name, geometry in geofabrik.map_regions(index):
            features.append(
                {
                    "type": "Feature",
                    "properties": {"id": region_id, "name": name},
                    "geometry": geometry.__geo_interface__,
                }
            )
        _regions_json = json.dumps({"type": "FeatureCollection", "features": features})
    return Response(_regions_json, mimetype="application/json")


@app.post("/api/plan")
def plan_area():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        bbox = _parse_bbox(payload)
        report = geofabrik.plan(bbox, CACHE_DIR, REPO_ROOT, _water_extra(payload))
    except (ValueError, OSError) as error:
        # A rectangle that is wrong stays wrong, so only a failure that came
        # from the network is worth offering to the user again.
        return jsonify({"error": str(error), "retryable": isinstance(error, OSError)}), 400
    if not report["regions"]:
        return jsonify({"error": "no published region covers this area"}), 404
    return jsonify(report)


@app.post("/api/generate")
def generate():
    payload = request.get_json(force=True, silent=True) or {}
    with _lock:
        if any(job["state"] == "running" for job in _jobs.values()):
            return jsonify({"error": "a job is already running"}), 409
        try:
            name = (payload.get("name") or "").strip() or time.strftime("map-%Y%m%d-%H%M%S")
            output = (payload.get("output") or "").strip() or f"out/{name}"
            features = _existing_file("features JSON", (payload.get("features") or DEFAULT_FEATURES).strip())
            zoom_min = int(payload.get("zoom_min", 6))
            zoom_max = int(payload.get("zoom_max", 17))
            if not ZOOM_MIN <= zoom_min <= zoom_max <= ZOOM_MAX:
                raise ValueError(f"zoom range must satisfy {ZOOM_MIN} <= min <= max <= {ZOOM_MAX}")
            bbox = _parse_bbox(payload) if payload.get("bbox") else None
            input_pbf = None
            if bbox is None:
                input_pbf = _existing_file("input PBF", (payload.get("input") or "").strip())
            water = bool(payload.get("water"))
            route = bool(payload.get("route"))
            # The route pack has a name of its own, named after the clock when
            # the field is left empty, the way the map does.
            route_name = (payload.get("route_name") or "").strip() if route else ""
            if route:
                route_name = route_name or time.strftime("route-%Y%m%d-%H%M%S")
                if not PLAIN_NAME.match(route_name):
                    raise ValueError("route name may only carry letters, digits, spaces, "
                                     "dots, dashes and underscores")
        except (ValueError, TypeError) as error:
            return jsonify({"error": str(error)}), 400

        output_dir = _repo_path(output)
        navmap_dir = output_dir / "NAVMAP"
        route_dir = output_dir / "ROUTE"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        log_path = LOG_DIR / f"{job_id}.log"

        job = {
            "id": job_id,
            "name": name,
            "route_name": route_name or None,
            "state": "running",
            "error": None,
            "output_dir": output_dir,
            "navmap_dir": navmap_dir,
            "route_dir": route_dir,
            "log_path": log_path,
            "started": time.time(),
            "finished": None,
            "bbox": bbox,
            "route": route,
            "cancel": threading.Event(),
            "progress": Progress(zoom_min, zoom_max),
            "pipeline": None,
            "downloaded": set(),
            "downloading": None,
            "cleaned": False,
        }
        spec = {
            "bbox": bbox,
            "input": input_pbf,
            "features": str(features),
            "zoom": [zoom_min, zoom_max],
            "water": water,
            "route": route,
        }
        _jobs[job_id] = job
        threading.Thread(target=_run_job, args=(job, spec), daemon=True).start()
        return jsonify(
            {
                "job": job_id,
                # The page names the map, never the folder it is written in.
                "name": name,
                "route_name": job["route_name"],
                "output": _relative(output_dir),
                "navmap": _relative(navmap_dir),
                "route": _relative(route_dir) if spec["route"] else None,
            }
        ), 202


@app.get("/api/job/<job_id>")
def job_status(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(
        {
            "id": job["id"],
            "name": job["name"],
            "route_name": job["route_name"],
            "state": job["state"],
            "error": job["error"],
            "output": None if job["cleaned"] else _relative(job["output_dir"]),
            "navmap": None if job["cleaned"] else _relative(job["navmap_dir"]),
            "route": None if job["cleaned"] or not job["route_dir"].exists()
            else _relative(job["route_dir"]),
            "cleaned": job["cleaned"],
            "progress": job["progress"].snapshot(),
            "stats": _stats(job) if job["state"] == "done" else None,
            "log": _log_lines(job["log_path"]),
        }
    )


@app.post("/api/jobs/clear")
def clear_packs():
    """Start over: the packs of the jobs that are over are deleted, the way a
    download would have deleted them, and the page takes back their buttons.

    Nothing is touched while a chain is writing its folder, and nothing is
    touched while an archive is being streamed out of it: deleting the pack of a
    download in flight would break it under the user's feet.  A job that is left
    alone is named in the answer, so what was kept is not a mystery.
    """
    cleared, kept = [], []
    with _lock:
        for job in _jobs.values():
            if job["state"] == "running" or job["cleaned"]:
                continue
            if job["downloading"]:
                kept.append(job["id"])
                continue
            shutil.rmtree(job["output_dir"], ignore_errors=True)
            job["cleaned"] = True
            cleared.append(job["id"])
    return jsonify({"cleared": cleared, "kept": kept})


@app.post("/api/job/<job_id>/cancel")
def job_cancel(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    if job["state"] != "running":
        return jsonify({"error": f"the job is already {job['state']}"}), 409
    # The event stops the chain between steps and inside a download; the signal
    # stops whatever command is running right now.
    job["cancel"].set()
    if job["pipeline"] is not None:
        job["pipeline"].stop()
    return jsonify({"state": "cancelling"})


@app.get("/api/job/<job_id>/archive")
def job_archive(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    part = request.args.get("part", "navmap")
    if part == "navmap":
        directory, arcname = job["navmap_dir"], "NAVMAP"
        filename = f'{job["name"]}-NAVMAP.zip'
    elif part == "route":
        directory, arcname = job["route_dir"], "ROUTE"
        # The route pack is saved with the name of the route, which is a field
        # of its own on the page (and `route-YYYYMMDD-HHMMSS` by default).
        filename = f'{job["route_name"] or job["name"] + "-ROUTE"}.zip'
    else:
        return jsonify({"error": "part must be navmap or route"}), 400
    if job["cleaned"]:
        return jsonify({"error": f"{arcname} was already downloaded and removed from disk"}), 409
    if not directory.is_dir():
        return jsonify({"error": f"{arcname} is not ready"}), 409

    def stream():
        # While this runs the folder cannot be cleared from under it.  A download
        # that the client abandons closes this generator at the yield, so the
        # flag has to come down on both ends (and the archives are only marked as
        # taken when they were actually read to the end).
        job["downloading"] = part
        try:
            yield from archive.stream_zip(directory, arcname)
        finally:
            job["downloading"] = None
        _downloaded(job, part)

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(stream(), mimetype="application/zip", headers=headers)


def _downloaded(job, part):
    """Delete the output of a job once every part of it has been downloaded.

    The archives are streamed, so when this runs the client already has the
    bytes.  Nothing is removed while some part is still waiting for its
    download: a job that made ROUTE keeps its folder until both are taken.
    """
    job["downloaded"].add(part)
    wanted = {"navmap"} | ({"route"} if job["route_dir"].is_dir() else set())
    if not wanted <= job["downloaded"]:
        return
    shutil.rmtree(job["output_dir"], ignore_errors=True)
    job["cleaned"] = True


def clear_outputs(out_dir):
    """Start with an empty out/: no pack from a previous service run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return out_dir


if __name__ == "__main__":
    print(f"out/ starts empty: {clear_outputs(REPO_ROOT / 'out')}")
    app.run(host="0.0.0.0", port=8080, threaded=True)
