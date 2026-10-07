/* Map view: Natural Earth base map with the Geofabrik regions highlighted, a
   rectangle drawn by dragging and zoom to reach regions that are small at world
   scale.  Antarctica is left out of the base map and out of reach, so the whole
   world fits at the least zoom with nothing painted below 60 S. */

const map = L.map('map', {
  zoomControl: true,
  attributionControl: false,
  scrollWheelZoom: true,
  doubleClickZoom: false,
  boxZoom: false,
  touchZoom: false,
  minZoom: 2,
  maxZoom: 12,
  maxBounds: [[-58, -180], [84, 180]],
  maxBoundsViscosity: 1.0,
});
map.setView([14, 0], 2);

let rectangle = null;
let startLatLng = null;
let dragging = false;
let drawMode = false;
let bounds = null;
let planTimer = null;
let jobTimer = null;

const elements = {
  rectangle: document.getElementById('rectangle'),
  bbox: document.getElementById('bbox'),
  plan: document.getElementById('plan'),
  retry: document.getElementById('retry'),
  form: document.getElementById('form'),
  name: document.getElementById('name'),
  features: document.getElementById('features'),
  zoomMin: document.getElementById('zoom_min'),
  zoomMax: document.getElementById('zoom_max'),
  route: document.getElementById('route'),
  routeName: document.getElementById('route_name'),
  routeNameLabel: document.getElementById('routename'),
  water: document.getElementById('water'),
  mode: document.getElementById('mode'),
  clear: document.getElementById('clear'),
  generate: document.getElementById('generate'),
  cancel: document.getElementById('cancel'),
  retryJob: document.getElementById('retryjob'),
  status: document.getElementById('status'),
  progress: document.getElementById('progress'),
  progressLabel: document.getElementById('progresslabel'),
  progressZoom: document.getElementById('progresszoom'),
  progressKey: document.getElementById('progresskey'),
  progressDetail: document.getElementById('progressdetail'),
  progressFill: document.getElementById('progressfill'),
  result: document.getElementById('result'),
  stats: document.getElementById('stats'),
};

/* The status line says what became of the job in the colour of that state. */
function setStatus(text, state) {
  elements.status.textContent = text;
  elements.status.className = state;
}

function human(bytes) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} kB`;
  return `${bytes} B`;
}

function humanSeconds(seconds) {
  if (seconds < 60) return `${seconds} s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

function currentBbox() {
  const west = Math.max(-180, bounds.getWest());
  const east = Math.min(180, bounds.getEast());
  const south = Math.max(-85, bounds.getSouth());
  const north = Math.min(85, bounds.getNorth());
  return [west, south, east, north];
}

function bboxText([west, south, east, north]) {
  return `${west.toFixed(4)}, ${south.toFixed(4)}  ->  ${east.toFixed(4)}, ${north.toFixed(4)}`;
}

function showBounds() {
  elements.rectangle.textContent = bboxText(currentBbox());
  elements.bbox.hidden = false;
}

/* The map moves like any other map; drawing is a mode of its own, so the hand
   that moves the view is never the one that draws.  Shift+drag draws from
   either mode, and the map goes back to moving once the area is picked.

   The button says the same thing whether or not the map is waiting: the area
   is picked by dragging and the drag ends by itself on release, so there is no
   drawing to stop and nothing else for it to be called.  Waiting shows as the
   button lit and the map under the crosshair, and pressing it again calls the
   drawing off. */

function setDrawMode(enabled) {
  drawMode = enabled;
  elements.mode.classList.toggle('active', enabled);
  document.getElementById('map').classList.toggle('drawing', enabled);
  if (enabled) map.dragging.disable();
  else map.dragging.enable();
}

elements.mode.addEventListener('click', () => setDrawMode(!drawMode));

/* Start over: no rectangle, no preview, no highlight, no mark of a job.

   The packs of the job are part of what is left over, so `Clear area` starts the
   service over as well: it asks for the packs of the jobs that are over to be
   deleted — the state a download would have left them in — and takes back what
   was on offer, the buttons of the packs and the box of figures.  Nothing is
   taken back while the service keeps a pack (it is being streamed to the browser)
   or while the request fails: a button for a pack that is still on disk has to
   stay.  The form is left alone; it is not a thing the area carried. */

function clearOffer() {
  elements.result.textContent = '';
  elements.stats.textContent = '';
  elements.stats.hidden = true;
}

async function clearPacks() {
  if (!elements.result.children.length && elements.stats.hidden) return;
  try {
    const response = await fetch('/api/jobs/clear', { method: 'POST' });
    if (!response.ok) return;
    const body = await response.json();
    // A pack the service kept is still on disk, so its button and its figures
    // stay where they are; with nothing kept there is nothing left to offer.
    if (body.kept && body.kept.length) return;
    clearOffer();
    if (body.cleared && body.cleared.length) setStatus('Packs deleted', 'done');
  } catch (error) {
    // The packs are still on disk, and still on offer.
  }
}

elements.clear.addEventListener('click', () => {
  if (rectangle) map.removeLayer(rectangle);
  rectangle = null;
  bounds = null;
  clearJobArea();
  highlightRegions([]);
  elements.rectangle.textContent = 'No area selected yet';
  elements.bbox.hidden = true;
  setPlanMessage('Draw an area to see what it downloads.', 'plan', false);
  setDrawMode(false);
  clearPacks();
});

map.on('mousedown', (event) => {
  if (!drawMode && !event.originalEvent.shiftKey) return;
  startLatLng = event.latlng;
  dragging = true;
  map.dragging.disable();
  if (rectangle) map.removeLayer(rectangle);
  rectangle = L.rectangle(L.latLngBounds(startLatLng, startLatLng), {
    color: '#4f9bf0',
    weight: 1.5,
    fillColor: '#4f9bf0',
    fillOpacity: 0.15,
    interactive: false,
  }).addTo(map);
});

map.on('mousemove', (event) => {
  if (dragging) rectangle.setBounds(L.latLngBounds(startLatLng, event.latlng));
});

function finishRectangle(event) {
  if (!dragging) return;
  dragging = false;
  if (!drawMode) map.dragging.enable();
  bounds = L.latLngBounds(startLatLng, event.latlng || startLatLng);
  if (bounds.getNorth() - bounds.getSouth() < 0.01 || bounds.getEast() - bounds.getWest() < 0.01) {
    map.removeLayer(rectangle);
    rectangle = null;
    bounds = null;
    elements.rectangle.textContent = 'Area too small, drag again';
    elements.bbox.hidden = true;
    highlightRegions([]);
    return;
  }
  showBounds();
  schedulePlan();
  setDrawMode(false);
}

map.on('mouseup', finishRectangle);
document.addEventListener('mouseup', (event) => {
  if (dragging) finishRectangle({ latlng: map.mouseEventToLatLng(event) });
});

/* The base map is the public domain Natural Earth land with its national borders
   in a solid line and its regions and provinces in a dashed one; land and
   country lines are always there, regions only from the zoom Natural Earth says
   each one is worth drawing at.  The base map lives in its own pane under the
   overlay pane, so a rectangle or a highlight is never hidden by it. */

map.createPane('basemap').style.zIndex = 350;

const basemapStyle = {
  land: { color: '#2f5f88', weight: 0.6, fillColor: '#173c5c', fillOpacity: 0.85 },
  country: { color: '#9dc0e4', weight: 1, fill: false },
  region: { color: '#5f8cba', weight: 0.8, dashArray: '4 3', fill: false },
};

const regionGroups = [];

function showRegions() {
  const zoom = map.getZoom();
  for (const [minZoom, group] of regionGroups) {
    if (zoom >= minZoom && !map.hasLayer(group)) group.addTo(map);
    else if (zoom < minZoom && map.hasLayer(group)) map.removeLayer(group);
  }
}

fetch('/static/basemap.geojson')
  .then((response) => (response.ok ? response.json() : Promise.reject(response.statusText)))
  .then((geojson) => {
    const outlines = geojson.features.filter((feature) => feature.properties.layer !== 'region');
    L.geoJSON({ type: 'FeatureCollection', features: outlines }, {
      pane: 'basemap',
      interactive: false,
      style: (feature) => basemapStyle[feature.properties.layer] || {},
    }).addTo(map);
    const byZoom = new Map();
    for (const feature of geojson.features) {
      if (feature.properties.layer !== 'region') continue;
      const zoom = feature.properties.min_zoom;
      if (!byZoom.has(zoom)) byZoom.set(zoom, []);
      byZoom.get(zoom).push(feature);
    }
    for (const [zoom, features] of byZoom) {
      regionGroups.push([zoom, L.geoJSON({ type: 'FeatureCollection', features }, {
        pane: 'basemap',
        interactive: false,
        style: basemapStyle.region,
      })]);
    }
    showRegions();
  })
  .catch(() => {});

let regionById = {};
let highlight = null;

fetch('/api/regions')
  .then((response) => (response.ok ? response.json() : Promise.reject(response.statusText)))
  .then((geojson) => {
    for (const feature of geojson.features) regionById[feature.properties.id] = feature;
  })
  .catch(() => {});

/* The Geofabrik polygons are the regions as they are downloaded, simplified for
   a map of the whole world, and their borders carry the territorial waters.  A
   line drawn from them looks like a frontier and is not one: several km off at
   high zoom.  Only the area is painted, and the paint goes away once the map is
   close enough for that drift to be seen through, so the lines on the map are
   always the Natural Earth ones. */

function highlightRegions(ids) {
  if (highlight) map.removeLayer(highlight);
  highlight = null;
  const features = ids.map((id) => regionById[id]).filter(Boolean);
  if (!features.length) return;
  highlight = L.geoJSON({ type: 'FeatureCollection', features }, {
    style: { stroke: false, fillColor: '#4f9bf0', fillOpacity: 0.3 },
    interactive: false,
  }).addTo(map);
  fitHighlight();
}

function fitHighlight() {
  if (highlight) highlight.setStyle({ fillOpacity: map.getZoom() >= 9 ? 0 : 0.3 });
}

map.on('zoomend', () => {
  fitHighlight();
  showRegions();
});

fetch('/api/config')
  .then((response) => response.json())
  .then((config) => {
    elements.features.value = config.features;
    elements.zoomMin.value = config.zoom_min;
    elements.zoomMax.value = config.zoom_max;
    elements.water.title = config.water_ready
      ? 'Water polygons ready'
      : 'The water polygons (about 900 MB) are downloaded the first time';
  })
  .catch(() => {});

/* Preview what the rectangle needs before committing to the download. */

function schedulePlan() {
  clearTimeout(planTimer);
  planTimer = setTimeout(runPlan, 250);
}

/* One place writes the preview box, so the retry button can never be left
   behind by a message that a retry would not fix. */
function setPlanMessage(text, className, retryable) {
  elements.plan.textContent = text;
  elements.plan.className = className;
  elements.retry.hidden = !retryable;
}

async function runPlan() {
  if (!bounds) return;
  const bbox = currentBbox();
  setPlanMessage('Planning...', 'plan', false);
  try {
    const response = await fetch('/api/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bbox }),
    });
    const body = await response.json();
    if (!response.ok) {
      highlightRegions([]);
      setPlanMessage(body.error || response.statusText, 'plan bad', body.retryable === true);
      return;
    }
    renderPlan(body);
  } catch (error) {
    setPlanMessage(`Planning failed: ${error}`, 'plan bad', true);
  }
}

/* The same rectangle, asked again: only offered when the failure was the
   network, never when the rectangle itself is the problem. */
elements.retry.addEventListener('click', () => {
  runPlan();
});

/* The plan is a list of lines: the figures of the area and one line per region,
   with a chip of the colour of what is about to happen to it.  The numbers are
   the ones the API returned, without a rounding of the page. */

function planCell(text, className) {
  const cell = document.createElement('span');
  cell.className = className;
  cell.textContent = text;
  return cell;
}

function planRow(cells, className) {
  const line = document.createElement('div');
  line.className = className ? `prow ${className}` : 'prow';
  line.append(...cells);
  return line;
}

function planRegion(region) {
  let state = 'download';
  if (region.stale) state = 'refresh';
  else if (region.cached) state = 'cached';
  const cells = [
    planCell(state, `chip ${state}`),
    planCell(region.name, 'regionname'),
    planCell(human(region.size), 'regionsize'),
  ];
  if (region.age_days !== null) cells.push(planCell(`${region.age_days.toFixed(1)} days old`, 'regionage'));
  if (region.probe_failed) cells.push(planCell('check failed', 'regionwarn'));
  return planRow(cells, 'region');
}

function renderPlan(plan) {
  highlightRegions(plan.regions.map((region) => region.id));
  const eta = plan.download_seconds ? ` (about ${humanSeconds(plan.download_seconds)})` : '';
  const body = [
    planRow([planCell('coverage', 'plankey'),
             planCell(`${(plan.coverage * 100).toFixed(1)}%`, 'planval')]),
    planRow([planCell('download', 'plankey'),
             planCell(`${human(plan.download_bytes)} of ${human(plan.total_bytes)}${eta}`, 'planval')]),
    planRow([planCell('disk', 'plankey'),
             planCell(`needs ${human(plan.required_bytes)}, free ${human(plan.free_bytes)}`, 'planval')]),
    ...plan.regions.map(planRegion),
  ];
  if (plan.deletes.length) {
    body.push(planRow([planCell('delete', 'plankey'),
                       planCell(`${human(plan.frees_bytes)} of cache from another area`, 'planval')]));
    for (const item of plan.deletes) {
      const mark = item.partial ? 'partial' : 'other';
      body.push(planRow([planCell(mark, `chip ${mark}`), planCell(item.name, 'regionname'),
                         planCell(human(item.size), 'regionsize')], 'region'));
    }
  }
  elements.plan.replaceChildren(...body);
  elements.plan.className = plan.fits ? 'plan ok' : 'plan bad';
  elements.retry.hidden = true;
}

/* The area of the job stays on the map: the rectangle that was drawn to pick it
   leaves when the job starts and this one takes its place, in the colour of the
   state, so the map says what is being generated and what came of it. */

const JOB_COLORS = {
  running: '#ffc266',
  done: '#6fb3f5',
  cancelled: '#9aa7b6',
  failed: '#ff8a8a',
};

const JOB_WORDS = {
  running: 'Generating',
  done: 'Generated',
  cancelled: 'Cancelled',
  failed: 'Failed',
};

let jobArea = null;
let jobRectangle = null;

function clearJobArea() {
  if (jobRectangle) map.removeLayer(jobRectangle);
  jobRectangle = null;
  jobArea = null;
}

function showJobArea(state) {
  if (!jobArea || !jobArea.bbox) return;
  const color = JOB_COLORS[state] || JOB_COLORS.cancelled;
  if (!jobRectangle) {
    const [west, south, east, north] = jobArea.bbox;
    jobRectangle = L.rectangle([[south, west], [north, east]], {
      weight: 2,
      fill: true,
      fillColor: color,
      fillOpacity: 0.07,
      interactive: false,
    }).addTo(map);
    jobRectangle.bindTooltip('', { permanent: true, direction: 'center', className: 'joblabel' });
  }
  jobRectangle.setStyle({ color, fillColor: color, dashArray: state === 'running' ? '7 5' : null });
  jobRectangle.setTooltipContent(`${JOB_WORDS[state] || state} ${jobArea.label}`);
}

/* Run the chain and follow its progress.  The payload is kept as it was sent,
   so a job that failed can be asked for again without redrawing anything. */

let currentJob = null;
let lastPayload = null;

async function startJob(payload) {
  lastPayload = payload;
  elements.generate.disabled = true;
  clearOffer();
  elements.progress.hidden = true;
  setStatus('Starting...', 'running');
  elements.cancel.hidden = true;
  elements.retryJob.hidden = true;
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      setStatus(body.error || response.statusText, 'failed');
      elements.generate.disabled = false;
      return;
    }
    currentJob = body.job;
    setStatus('Running', 'running');
    elements.cancel.hidden = false;
    elements.cancel.disabled = false;
    // The area is picked: the rectangle of the drawing hands the map over to
    // the job, which paints it in the colour of its state.
    if (rectangle) map.removeLayer(rectangle);
    rectangle = null;
    jobArea = { bbox: payload.bbox || null, label: body.name };
    showJobArea('running');
    clearInterval(jobTimer);
    jobTimer = setInterval(() => poll(body.job), 1000);
    poll(body.job);
  } catch (error) {
    setStatus(`Request failed: ${error}`, 'failed');
    elements.generate.disabled = false;
  }
}

/* The route pack has a name of its own: the field only exists while the routes
   are asked for, and empty means the service names it after the clock
   (`route-YYYYMMDD-HHMMSS`), the way it names a map that was left unnamed. */
function syncRouteName() {
  const wanted = elements.route.checked;
  elements.routeNameLabel.hidden = !wanted;
  elements.routeName.hidden = !wanted;
}

elements.route.addEventListener('change', syncRouteName);
syncRouteName();

elements.form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!bounds) {
    setStatus('Draw an area on the map first', 'failed');
    return;
  }
  startJob({
    name: elements.name.value.trim(),
    route_name: elements.route.checked ? elements.routeName.value.trim() : '',
    features: elements.features.value.trim(),
    zoom_min: elements.zoomMin.value,
    zoom_max: elements.zoomMax.value,
    route: elements.route.checked,
    water: elements.water.checked,
    bbox: currentBbox(),
  });
});

/* Stop the job: the service kills the running command and marks it cancelled. */
elements.cancel.addEventListener('click', async () => {
  if (!currentJob) return;
  elements.cancel.disabled = true;
  setStatus('Cancelling...', 'running');
  try {
    await fetch(`/api/job/${currentJob}/cancel`, { method: 'POST' });
  } catch (error) {
    setStatus(`Cancel failed: ${error}`, 'failed');
  }
});

/* Once more: the payload of the last run, not what the form says now, so a
   rectangle that was cleared meanwhile cannot change what is generated. */
elements.retryJob.addEventListener('click', () => {
  if (lastPayload) startJob(lastPayload);
});

async function poll(job) {
  const response = await fetch(`/api/job/${job}`);
  if (!response.ok) {
    setStatus('job lost', 'failed');
    clearInterval(jobTimer);
    elements.generate.disabled = false;
    elements.cancel.hidden = true;
    elements.progress.hidden = true;
    return;
  }
  const body = await response.json();
  updateProgress(body.state, body.progress);
  showJobArea(body.state);
  if (body.state === 'running') return;
  clearInterval(jobTimer);
  currentJob = null;
  elements.generate.disabled = false;
  elements.cancel.hidden = true;
  elements.retryJob.hidden = body.state === 'done' || !lastPayload;
  if (body.cleaned) {
    setStatus('Downloaded; the job folder is empty again', 'done');
  } else if (body.state === 'done') {
    setStatus('Done', 'done');
    renderResult(body);
    renderStats(body.stats);
  } else if (body.state === 'cancelled') {
    setStatus('Cancelled', 'cancelled');
  } else {
    setStatus(body.error ? `Failed -> ${body.error}` : 'Failed', 'failed');
  }
}

/* The bar of the step that is running.  The chain reports how far along it is in
   two ways: a download counts bytes, and the tile generator counts the tiles of
   the zoom it is building, which is the only step that knows its own length.  A
   step with no number of its own (a merge, a clip) shows the bar as working,
   never as a fraction the page made up, and says how long it has been running. */

let stepKey = null;
let stepStarted = 0;

function updateProgress(state, progress) {
  const alive = state === 'running' && progress && progress.phase !== 'idle';
  elements.progress.hidden = !alive;
  if (!alive) {
    stepKey = null;
    return;
  }
  // The clock of a step starts when the page first sees it, so a step that
  // reports nothing still says something that was measured.
  const key = `${progress.phase}:${progress.label}`;
  if (key !== stepKey) {
    stepKey = key;
    stepStarted = Date.now();
  }
  elements.progressLabel.textContent = progress.label || progress.phase;
  const known = progress.percent !== null;
  elements.progress.classList.toggle('unknown', !known);
  elements.progressFill.style.width = known ? `${progress.percent}%` : '';
  elements.progressKey.hidden = known;
  elements.progressDetail.textContent = known
    ? progressNumbers(progress)
    : humanElapsed(Math.floor((Date.now() - stepStarted) / 1000));
  // The zoom being built is named beside the step, where it was: `zoom 14/17`.
  // It is the only count of the whole job that is known while the tiles are
  // built: the tiles of the zooms still to come are not known until each one
  // starts, so a bar for the whole job would have to guess them, and it is not
  // drawn.
  const zooming = progress.zoom !== null;
  elements.progressZoom.hidden = !zooming;
  elements.progressZoom.textContent = zooming ? `zoom ${progress.zoom}/${progress.zoom_max}` : '';
}

/* How long a step has been running, by the page's own clock.  The service
   measures in tenths of a second, so a figure over a minute keeps its tenth:
   the rest of the minute is rounded the same way, because the remainder of a
   binary subtraction is not (`2 min 31.30000000000001 s` was the merge of a
   real job). */
function humanElapsed(seconds) {
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ${Math.round((seconds % 60) * 10) / 10} s`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

/* What the bar of the step reports: the percentage, and the tiles when the step
   counts tiles (a download counts bytes).  The rate the engine prints is left
   out, it says nothing the counter does not. */
function progressNumbers(progress) {
  if (!progress.total) return `${progress.percent}%`;
  if (progress.phase === 'download') {
    return `${progress.percent}% · ${human(progress.done)} of ${human(progress.total)}`;
  }
  return `${progress.percent}% · ${progress.done}/${progress.total} tiles`;
}

/* What the job did, once it is over: the time it took and, under it, where that
   time went step by step; the tiles the engine counted while it built the zooms;
   and what the packs weigh.  Every figure is one the service measured, and a job
   with no routes has no line of routes. */
function statRow(key, value, className) {
  const row = document.createElement('div');
  row.className = className ? `prow ${className}` : 'prow';
  const name = document.createElement('span');
  name.className = 'plankey';
  name.textContent = key;
  const figure = document.createElement('span');
  figure.className = 'planval';
  figure.textContent = value;
  row.append(name, figure);
  return row;
}

function renderStats(stats) {
  if (!stats) return;
  const rows = [statRow('Total time', humanElapsed(stats.seconds))];
  for (const step of stats.steps) rows.push(statRow(step.name, humanElapsed(step.seconds), 'sub'));
  // The tiles and the area are written plain: a thousands separator is not used
  // anywhere else in the panel.
  rows.push(statRow('Tiles', `${stats.tiles}`));
  if (stats.navmap_bytes !== null) rows.push(statRow('Map size', human(stats.navmap_bytes)));
  if (stats.route_bytes !== null) rows.push(statRow('Route size', human(stats.route_bytes)));
  if (stats.area_km2 !== null) rows.push(statRow('Area', `${stats.area_km2} km²`));
  if (stats.bbox) rows.push(statRow('BBox', bboxText(stats.bbox)));
  elements.stats.replaceChildren(...rows);
  elements.stats.hidden = false;
}

/* The service deletes the folder of the job once every part of it has been
   downloaded, which happens after the link was pressed.  Keep an eye on the
   job until then so the page says what became of the packs. */
let cleanupTimer = null;

function followCleanup(jobId) {
  clearInterval(cleanupTimer);
  let ticks = 0;
  cleanupTimer = setInterval(async () => {
    ticks += 1;
    const response = await fetch(`/api/job/${jobId}`).catch(() => null);
    if (!response || !response.ok) {
      clearInterval(cleanupTimer);
      return;
    }
    const body = await response.json();
    if (body.cleaned) {
      clearInterval(cleanupTimer);
      setStatus('Downloaded; the job folder is empty again', 'done');
      // The packs are gone with their figures, which were of that job.
      clearOffer();
    } else if (ticks > 240) {
      clearInterval(cleanupTimer);
    }
  }, 5000);
}

/* What the job leaves behind is downloaded from here, named by what it is and
   not by the file it will become: the folder the service writes in is its own
   business, and the name of the map is already on the map. */
function renderResult(body) {
  elements.result.textContent = '';
  const parts = [['map file', 'navmap']];
  if (body.route) parts.push(['route file', 'route']);
  for (const [label, part] of parts) {
    const link = document.createElement('a');
    link.className = 'download';
    link.href = `/api/job/${body.id}/archive?part=${part}`;
    link.textContent = `Download ${label}`;
    link.addEventListener('click', () => followCleanup(body.id));
    elements.result.appendChild(link);
  }
}
