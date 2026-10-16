# Web Service

`web/` is a Flask service that serves one page: pick an area on a map and get the packs for it without
typing an `osmium` command. It runs the same chain as the CLI — download the Geofabrik PBFs that cover
the rectangle into `cache/`, merge them, drop the versions the extracts share at their borders
(`osmium time-filter`, or the merged file has a repeated ID and `osmium extract` refuses it), clip with
`osmium extract`, run `nav_generator` (and `route_generator` when routes are asked for) — and writes
`out/<name>/NAVMAP` and `ROUTE`. The generators are untouched: the service only writes their commands
and reads their output.

---

## Running it

```bash
docker compose build web          # once, or whenever the Dockerfile changes
docker compose up -d web          # serves the page on 127.0.0.1:8080 (loopback only)
docker compose logs -f web        # follow the service log
docker compose restart web        # after editing web/*.py
docker compose down
```

Without `-d` the log stays in the terminal and Ctrl+C stops the service. A change under `web/static/`
needs no restart: the browser picks it up on reload. The container reads the host timezone through
`/etc/localtime`, so the default `map-YYYYMMDD-HHMMSS` name matches your clock. If the daemon answers
`permission denied while trying to connect to the Docker daemon socket`, run `newgrp docker` once, or
prefix the commands with `sg docker -c "..."`.

---

## The page, in order

1. **Draw the area** with `Draw area` or `shift`+drag; under 0.01° is refused. `Clear area` starts over.
2. **Plan** looks at what the rectangle needs and touches nothing: the regions that cover it, whether
   each one is cached, to download or over a week old, the MB that travel, how long that takes at the
   measured 5 MB/s, and whether the disk can hold the result. More than **12 regions**, or none, is
   refused here.
3. **Generate** runs the chain, one job at a time (a second one gets a `409`). The bar follows the step
   that is running with its own numbers; a step that prints none shows `elapsed` with its time.
4. **Download** streams the ZIPs while they are built — `<name>-NAVMAP.zip`, and `<name>.zip` from the
   `Route name` field — and deletes the job folder once they are out. A job in `Done` reports its total
   time, that time step by step, the tiles the engine counted, what the packs weigh and the area.
5. The area of the job stays marked on the map in the colour of its state (`Running`, `Done`,
   `Cancelled`, `Failed`), with a label naming the pack, and a finished job is **previewed on the map**
   over the rectangle through `Hide preview` / `Show preview`, until the packs are downloaded.

## `cache/`, and what survives

- The Geofabrik PBFs stay between jobs, so only the missing regions travel. The index of regions
  refreshes once a day, and a failed refresh keeps the cached copy. Sizes and dates come from a one-byte
  `Range: bytes=0-0` probe: the `-latest` URLs no longer answer `HEAD`.
- A copy older than **7 days** is replaced when Geofabrik has published something newer; if the probe
  does not answer, the copy is kept and the job goes on with it. `Generate` deletes the PBFs of another
  area, and the plan announces which ones and how many MB it frees. The water archive and the index are
  never deleted, and previewing never downloads or deletes anything.
- A cut download resumes with `Range` and leaves its `.part` behind; the `.part` of a region the area
  needs is kept. A region is only pulled when it covers at least 0.5% of the rectangle, and the disk
  guard asks for 10× the input plus two working copies, ×1.25.
- The packs land in `out/<name>/`, emptied when the service starts and gone once the ZIPs are
  downloaded, and the job log in `logs/<job>.log` — outside `out/`, so it survives the cleanup.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/config` | the defaults of the form |
| GET | `/api/regions` | the Geofabrik index as GeoJSON |
| POST | `/api/plan` | what the rectangle needs, without touching anything |
| POST | `/api/generate` | starts a job (`202`), one at a time |
| GET | `/api/job/<id>` | state, progress, figures and the tail of its log |
| POST | `/api/job/<id>/cancel` | kills the process; the job ends `cancelled` |
| GET | `/api/job/<id>/archive?part=navmap\|route` | streams the ZIP |
| GET | `/api/job/<id>/tile/<z>/<x>/<y>.png` | one tile of the preview, or `404` when there is none |
| POST | `/api/jobs/clear` | deletes the packs of the jobs that are over |

## When something goes wrong

- The log is `logs/<job>.log`; the page shows the bar and the state (`Running`, `Done`, `Cancelled`,
  `Failed`) and never the log.
- A plan that failed on the network offers `Retry`; a rectangle that is wrong does not, because asking
  again would not change the answer.
- `Cancel` kills the running command. `Retry job` resends the payload of the last run, not what the form
  says now.
- The water polygons (about 900 MB) are downloaded the first time the checkbox is used and clipped to the
  rectangle, as [`../WATER.md`](../WATER.md) describes. The base map is a vendored file that no setup step
  builds; see [`basemap.md`](basemap.md).
