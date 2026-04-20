"use strict";

const DEFAULT_LAT = 43.2567;
const DEFAULT_LON = 76.9286;

const ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const OSM_URL  = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";

let singleMap        = null;
let singleMarker     = null;
let batchMap         = null;
let batchPredictions = null;

// ── Lazy-init single map on expander open ─────────────────────
document.getElementById("map-expander").addEventListener("toggle", function () {
  if (!this.open) return;
  if (!singleMap) initSingleMap();
  else singleMap.invalidateSize();
});

function makeLayers() {
  return {
    esri: L.tileLayer(ESRI_URL, { attribution: "© Esri", maxZoom: 19 }),
    osm:  L.tileLayer(OSM_URL,  { attribution: "© OpenStreetMap contributors", maxZoom: 19 }),
  };
}

function addMousePos(map) {
  const MousePos = L.Control.extend({
    options: { position: "topright" },
    onAdd(m) {
      const div = L.DomUtil.create("div", "lf-mouse-pos leaflet-control");
      div.innerHTML = "&nbsp;";
      m.on("mousemove", e => {
        div.textContent = e.latlng.lat.toFixed(6) + "   " + e.latlng.lng.toFixed(6);
      });
      m.on("mouseout", () => { div.innerHTML = "&nbsp;"; });
      return div;
    },
  });
  new MousePos().addTo(map);
}

function addFullscreen(map) {
  const FSControl = L.Control.extend({
    options: { position: "topright" },
    onAdd() {
      const btn = L.DomUtil.create("button", "leaflet-bar leaflet-control lf-fs-btn");
      btn.title       = "Fullscreen";
      btn.textContent = "[ ]";
      L.DomEvent.disableClickPropagation(btn);
      L.DomEvent.on(btn, "click", () => {
        const el = map.getContainer();
        if (!document.fullscreenElement) el.requestFullscreen?.();
        else document.exitFullscreen?.();
      });
      return btn;
    },
  });
  new FSControl().addTo(map);
}

function initSingleMap() {
  const { esri, osm } = makeLayers();
  singleMap = L.map("single-map", {
    center: [DEFAULT_LAT, DEFAULT_LON],
    zoom: 12,
    layers: [esri],
    attributionControl: false,
  });
  L.control.attribution({ prefix: false, position: "bottomright" }).addTo(singleMap);
  L.control.layers({ "Satellite": esri, "Street Map": osm }, null, {
    collapsed: true, position: "topright",
  }).addTo(singleMap);
  addMousePos(singleMap);
  addFullscreen(singleMap);

  try {
    L.Control.geocoder({ position: "topleft", defaultMarkGeocode: false })
      .on("markgeocode", e => {
        const c = e.geocode.center;
        setLocation(c.lat, c.lng);
        singleMap.setView(c, 15);
      })
      .addTo(singleMap);
  } catch (_) {}

  singleMap.on("click", e => setLocation(e.latlng.lat, e.latlng.lng));
  singleMarker = L.circleMarker([DEFAULT_LAT, DEFAULT_LON], markerStyle()).addTo(singleMap);
}

// ── Coordinate sync ───────────────────────────────────────────
document.getElementById("LATITUDE").addEventListener("change",  syncMapFromInputs);
document.getElementById("LONGITUDE").addEventListener("change", syncMapFromInputs);

function syncMapFromInputs() {
  const lat = parseFloat(document.getElementById("LATITUDE").value);
  const lon = parseFloat(document.getElementById("LONGITUDE").value);
  if (isNaN(lat) || isNaN(lon) || !singleMap) return;
  singleMarker?.setLatLng([lat, lon]);
  singleMap.setView([lat, lon]);
}

function setLocation(lat, lon) {
  document.getElementById("LATITUDE").value  = lat.toFixed(6);
  document.getElementById("LONGITUDE").value = lon.toFixed(6);
  singleMarker?.setLatLng([lat, lon]);
}

// ── Single prediction ─────────────────────────────────────────
async function runPredict() {
  const payload = {
    ROOMS:        int_("ROOMS"),
    LONGITUDE:    float_("LONGITUDE"),
    LATITUDE:     float_("LATITUDE"),
    TOTAL_AREA:   float_("TOTAL_AREA"),
    FLOOR:        int_("FLOOR"),
    TOTAL_FLOORS: int_("TOTAL_FLOORS"),
    FURNITURE:    int_("FURNITURE"),
    CONDITION:    int_("CONDITION"),
    MATERIAL:     int_("MATERIAL"),
    YEAR:         int_("YEAR"),
  };
  if (isNaN(payload.LATITUDE) || isNaN(payload.LONGITUDE)) {
    showError("Enter valid coordinates or click on the map."); return;
  }
  setLoadingSingle(true);
  hide("error-card");
  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(e.detail);
    }
    const data = await resp.json();
    renderSingleResult(data, payload);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoadingSingle(false);
  }
}

function setLoadingSingle(on) {
  const btn = document.getElementById("predict-btn");
  btn.disabled    = on;
  btn.textContent = on ? "Calculating..." : "Predict Price";
  on ? show("loading-card") : hide("loading-card");
}

function renderSingleResult(data, p) {
  document.getElementById("price-banner").textContent = fmt(data.price_kzt) + " KZT";
  show("result-section");
  if (singleMarker) {
    singleMarker.unbindPopup();
    singleMarker.bindPopup(fmt(data.price_kzt) + " KZT").openPopup();
  }
  const COLS = [
    "ROOMS", "TOTAL_AREA", "FLOOR", "TOTAL_FLOORS", "FURNITURE", "CONDITION",
    "MATERIAL", "YEAR", "LATITUDE", "LONGITUDE", "pred_price_kzt",
  ];
  const row = { ...p, pred_price_kzt: data.price_kzt };
  let html = '<table class="data-table"><thead><tr>';
  COLS.forEach(c => { html += "<th>" + c + "</th>"; });
  html += "</tr></thead><tbody><tr>";
  COLS.forEach(c => {
    const v = row[c];
    html += "<td>" + (c === "pred_price_kzt" ? fmt(v) + " KZT" : v) + "</td>";
  });
  html += "</tr></tbody></table>";
  document.getElementById("details-table-wrap").innerHTML = html;
}

// ── Batch upload ──────────────────────────────────────────────
async function handleBatchUpload() {
  const file = document.getElementById("batch-file").files[0];
  if (!file) return;
  hide("batch-results");
  show("batch-loading");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const resp = await fetch("/batch", { method: "POST", body: fd });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(e.detail);
    }
    batchPredictions = await resp.json();
    renderBatchResults(batchPredictions);
    document.getElementById("dl-csv-btn").disabled  = false;
    document.getElementById("dl-xlsx-btn").disabled = false;
  } catch (err) {
    alert("Batch error: " + err.message);
  } finally {
    hide("batch-loading");
  }
}

function renderBatchResults(rows) {
  document.getElementById("batch-banner").textContent = "Predicted " + rows.length + " properties";
  const COLS = [
    "ROOMS", "TOTAL_AREA", "FLOOR", "TOTAL_FLOORS", "FURNITURE", "CONDITION",
    "MATERIAL", "YEAR", "LATITUDE", "LONGITUDE", "pred_price_kzt",
  ];
  let html = '<table class="data-table"><thead><tr>';
  COLS.forEach(c => { html += "<th>" + c + "</th>"; });
  html += "</tr></thead><tbody>";
  rows.slice(0, 50).forEach(row => {
    html += "<tr>";
    COLS.forEach(c => {
      const v = row[c];
      html += "<td>" + (c === "pred_price_kzt" && v != null ? fmt(v) : (v ?? "")) + "</td>";
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  document.getElementById("batch-table-wrap").innerHTML = html;
  show("batch-results");
}

// ── Batch map (lazy-init on expander open) ────────────────────
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("batch-map-expander").addEventListener("toggle", function () {
    if (!this.open) return;
    if (!batchMap) initBatchMap();
    else {
      batchMap.invalidateSize();
      if (batchPredictions) renderBatchMarkers();
    }
  });
});

function initBatchMap() {
  const { esri, osm } = makeLayers();
  batchMap = L.map("batch-map", {
    center: [DEFAULT_LAT, DEFAULT_LON],
    zoom: 10,
    layers: [esri],
    attributionControl: false,
  });
  L.control.attribution({ prefix: false, position: "bottomright" }).addTo(batchMap);
  L.control.layers({ "Satellite": esri, "Street Map": osm }, null, {
    collapsed: true, position: "topright",
  }).addTo(batchMap);
  addMousePos(batchMap);
  addFullscreen(batchMap);

  try {
    L.Control.geocoder({ position: "topleft", defaultMarkGeocode: false })
      .on("markgeocode", e => batchMap.setView(e.geocode.center, 14))
      .addTo(batchMap);
  } catch (_) {}

  if (batchPredictions) renderBatchMarkers();
}

const FUN_LBL  = { 1: "No", 2: "Partial", 3: "Full" };
const MAT_LBL  = { 1: "Panel", 2: "Brick", 3: "Monolithic", 4: "Mixed" };
const COND_LBL = { 1: "Needs renovation", 2: "Fair", 3: "Good", 4: "Excellent", 5: "Perfect" };

function renderBatchMarkers() {
  if (!batchMap || !batchPredictions) return;
  batchMap.eachLayer(l => { if (l instanceof L.CircleMarker) batchMap.removeLayer(l); });
  const pts = [];
  batchPredictions.forEach(row => {
    const lat = +row.LATITUDE, lon = +row.LONGITUDE;
    if (isNaN(lat) || isNaN(lon)) return;
    pts.push([lat, lon]);
    const priceStr = fmt(row.pred_price_kzt) + " KZT";
    const popup =
      '<div class="map-popup">' +
        '<div class="popup-price">' + priceStr + "</div>" +
        "<div>Coordinates: " + lat.toFixed(6) + ", " + lon.toFixed(6) + "</div>" +
        "<div>Rooms: " + row.ROOMS + "</div>" +
        "<div>Total area (m\u00b2): " + row.TOTAL_AREA + "</div>" +
        "<div>Floor/Total: " + row.FLOOR + "/" + row.TOTAL_FLOORS + "</div>" +
        "<div>Furniture: " + (FUN_LBL[row.FURNITURE] || row.FURNITURE) + "</div>" +
        "<div>Condition: " + (COND_LBL[row.CONDITION] || row.CONDITION) + "</div>" +
        "<div>Material: " + (MAT_LBL[row.MATERIAL] || row.MATERIAL) + "</div>" +
        "<div>Year: " + row.YEAR + "</div>" +
      "</div>";
    L.circleMarker([lat, lon], markerStyle())
      .bindPopup(popup)
      .bindTooltip(priceStr, { direction: "top", sticky: true })
      .addTo(batchMap);
  });
  if (pts.length > 0) batchMap.fitBounds(L.latLngBounds(pts).pad(0.1));
}

// ── Downloads ─────────────────────────────────────────────────
function downloadCSV() {
  if (!batchPredictions) return;
  const COLS = [
    "ROOMS", "TOTAL_AREA", "FLOOR", "TOTAL_FLOORS", "FURNITURE", "CONDITION",
    "MATERIAL", "YEAR", "LATITUDE", "LONGITUDE", "pred_price_kzt",
  ];
  let csv = COLS.join(",") + "\n";
  batchPredictions.forEach(r => { csv += COLS.map(c => r[c] ?? "").join(",") + "\n"; });
  trigger(new Blob([csv], { type: "text/csv" }), "predictions.csv");
}

async function downloadXLSX() {
  if (!batchPredictions) return;
  try {
    const resp = await fetch("/batch/download/xlsx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(batchPredictions),
    });
    trigger(await resp.blob(), "predictions.xlsx");
  } catch (e) {
    alert("Download failed: " + e.message);
  }
}

function trigger(blob, name) {
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob), download: name,
  });
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Helpers ───────────────────────────────────────────────────
function int_(id)   { return parseInt(document.getElementById(id).value, 10); }
function float_(id) { return parseFloat(document.getElementById(id).value); }
function fmt(n)     { return Math.round(n).toLocaleString("ru-RU"); }
function show(id)   { document.getElementById(id)?.classList.remove("hidden"); }
function hide(id)   { document.getElementById(id)?.classList.add("hidden"); }
function markerStyle() {
  return { radius: 6, color: "#00E0A8", fillColor: "#00E0A8", fillOpacity: 0.8, weight: 1.5 };
}
function showError(msg) {
  document.getElementById("error-msg").textContent = msg;
  show("error-card");
}
